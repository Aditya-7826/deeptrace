// DeepTrace content script. Injected only when the user clicks "Scan this page".
// Collects images, shows badges in a shadow-DOM overlay (page layout is never touched)
// and leaves all network work to the service worker.
(() => {
  "use strict";

  if (globalThis.__deeptraceContent) return; // already injected: listener is registered
  globalThis.__deeptraceContent = true;

  const { skipReason, normalizeUrl, UrlDeduper } = globalThis.DeepTraceFilter;
  const { badgeFor } = globalThis.DeepTraceLabels;

  const WATCH_MS = 2000; // lazy images that load within 2 s of scrolling are included
  const WATCH_CAP_MS = 60000;
  const PING_MS = 5000; // wakes (or restarts) the service worker while results are pending
  const RESULT_TIMEOUT_MS = 180000;
  const MAX_BYTES = 10 * 1024 * 1024;

  const deduper = new UrlDeduper();
  const entries = new Map(); // id -> {img, url, badge, result, state, since}
  const seenImages = new WeakSet();
  let nextId = 1;
  let showConfidence = true;
  let watchUntil = 0;
  let scanStarted = 0;
  let watchTimer = null;
  let pingTimer = null;

  // ---- overlay ---------------------------------------------------------------------

  const host = document.createElement("deeptrace-overlay");
  host.style.cssText =
    "all: initial; position: fixed; top: 0; left: 0; width: 0; height: 0; " +
    "z-index: 2147483647; pointer-events: none;";
  const root = host.attachShadow({ mode: "closed" });
  const style = document.createElement("style"); // no innerHTML: pages may enforce Trusted Types
  style.textContent = `
      :host { all: initial; }
      .badge {
        position: fixed; top: 0; left: 0; pointer-events: auto; cursor: pointer;
        font: 600 12px/1.2 system-ui, -apple-system, "Segoe UI", sans-serif;
        padding: 3px 7px; border-radius: 10px; border: 1px solid rgba(0,0,0,.25);
        color: #fff; background: #495057; box-shadow: 0 1px 3px rgba(0,0,0,.35);
        white-space: nowrap; max-width: 220px; overflow: hidden; text-overflow: ellipsis;
      }
      .badge:focus-visible { outline: 2px solid #ffd43b; outline-offset: 1px; }
      .badge[hidden] { display: none; }
      .real { background: #2b8a3e; }
      .fake { background: #c92a2a; }
      .uncertain { background: #e67700; }
      .no_face { background: #868e96; cursor: default; }
      .error { background: #5c5f66; }
      .pending { background: #3b5bdb; cursor: progress; }
      .banner {
        position: fixed; top: 12px; right: 12px; max-width: 380px; pointer-events: auto;
        font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: #212529;
        background: #fff; border: 1px solid #ced4da; border-left: 5px solid #c92a2a;
        border-radius: 8px; padding: 12px 14px; box-shadow: 0 4px 16px rgba(0,0,0,.2);
      }
      .banner.info { border-left-color: #3b5bdb; }
      .banner code { font: 12px ui-monospace, Consolas, monospace; background: #f1f3f5;
        padding: 1px 4px; border-radius: 4px; }
      .banner .row { display: flex; gap: 8px; margin-top: 10px; justify-content: flex-end; }
      .banner button { font: inherit; font-size: 13px; padding: 4px 10px; border-radius: 6px;
        border: 1px solid #adb5bd; background: #f8f9fa; color: #212529; cursor: pointer; }`;
  root.appendChild(style);
  document.documentElement.appendChild(host);

  function showBanner(text, { kind = "error", settings = false, timeoutMs = 0 } = {}) {
    root.querySelectorAll(".banner").forEach((b) => b.remove());
    const box = document.createElement("div");
    box.className = `banner ${kind === "info" ? "info" : ""}`;
    box.setAttribute("role", kind === "info" ? "status" : "alert");
    // `cmd` spans become <code>; everything else is text (never HTML)
    String(text).split("`").forEach((part, i) => {
      const node = document.createElement(i % 2 ? "code" : "span");
      node.textContent = part;
      box.appendChild(node);
    });
    const row = document.createElement("div");
    row.className = "row";
    if (settings) {
      const open = document.createElement("button");
      open.textContent = "Settings";
      open.addEventListener("click", () =>
        chrome.runtime.sendMessage({ type: "deeptrace:open-options" }));
      row.appendChild(open);
    }
    const close = document.createElement("button");
    close.textContent = "Close";
    close.addEventListener("click", () => box.remove());
    row.appendChild(close);
    box.appendChild(row);
    root.appendChild(box);
    if (timeoutMs) setTimeout(() => box.remove(), timeoutMs);
  }

  // ---- badges ----------------------------------------------------------------------

  function makeBadge(id) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "badge pending";
    b.textContent = "Scanning…";
    b.title = "DeepTrace is analysing this image";
    b.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      onBadgeClick(id);
    });
    root.appendChild(b);
    return b;
  }

  function render(entry) {
    const b = entry.badge;
    if (entry.state === "pending") return;
    if (entry.state === "explaining") {
      b.className = "badge pending";
      b.textContent = "Heatmap…";
      return;
    }
    const v = badgeFor(entry.result, showConfidence);
    b.className = `badge ${v.kind}`;
    b.textContent = v.text;
    b.title = v.title;
    b.setAttribute("aria-label", `DeepTrace: ${v.text}. ${v.title}`);
  }

  function place(entry) {
    const b = entry.badge;
    const img = entry.img;
    if (!img.isConnected) {
      b.hidden = true;
      return;
    }
    const r = img.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    const visible = r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 &&
      r.top < vh && r.left < vw;
    b.hidden = !visible;
    if (visible) {
      const x = Math.max(r.left, 0) + 4;
      const y = Math.max(r.top, 0) + 4;
      b.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    }
  }

  let frame = 0;
  function placeAll() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      for (const e of entries.values()) place(e);
    });
  }
  window.addEventListener("scroll", placeAll, { capture: true, passive: true });
  window.addEventListener("resize", placeAll, { passive: true });
  const resizeObserver = new ResizeObserver(placeAll);

  // ---- collecting images -----------------------------------------------------------

  function rendered(img) {
    if (!img.complete || !img.naturalWidth) return false;
    if (typeof img.checkVisibility === "function") {
      return img.checkVisibility({ opacityProperty: true, visibilityProperty: true });
    }
    const r = img.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  async function blobToDataUrl(url) {
    const blob = await (await fetch(url)).blob();
    if (blob.size > MAX_BYTES) throw new Error("the image is larger than 10 MB");
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  }

  /** Decides on one <img>. Unloaded images are left for their load event. */
  async function consider(img) {
    if (seenImages.has(img) || !rendered(img)) return null;
    seenImages.add(img);
    const r = img.getBoundingClientRect();
    const src = img.currentSrc || img.src;
    const reason = skipReason({ src, naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight, width: r.width, height: r.height });
    if (reason) return null;
    const url = normalizeUrl(src, document.baseURI);
    if (!deduper.add(url)) return null;

    const id = nextId++;
    const entry = { img, url, badge: makeBadge(id), result: null, state: "pending",
      since: Date.now(), sendUrl: url };
    entries.set(id, entry);
    resizeObserver.observe(img);
    place(entry);
    if (url.startsWith("blob:")) {
      // blob: URLs only resolve inside this page, so read the bytes here
      try {
        entry.sendUrl = await blobToDataUrl(url);
      } catch (e) {
        finish(id, "predict", { error: "fetch_failed", message: String(e.message || e) });
        return null;
      }
    } else if (url.startsWith("data:") && url.length * 0.75 > MAX_BYTES) {
      finish(id, "predict", { error: "file_too_large", message: "the image is larger than 10 MB" });
      return null;
    }
    return { id, url: entry.sendUrl };
  }

  async function collect(imgs) {
    const items = (await Promise.all([...imgs].map(consider))).filter(Boolean);
    if (items.length) {
      chrome.runtime.sendMessage({ type: "deeptrace:enqueue", items }).catch(() => {});
      startPinging();
    }
    return items.length;
  }

  // Lazy-loaded images: a 2 s window after the scan starts, extended by each scroll.
  function onImageLoad(ev) {
    if (Date.now() <= watchUntil && ev.target instanceof HTMLImageElement) collect([ev.target]);
  }
  function onScroll() {
    watchUntil = Math.min(Date.now() + WATCH_MS, scanStarted + WATCH_CAP_MS);
    armWatch();
  }
  function armWatch() {
    clearTimeout(watchTimer);
    const left = watchUntil - Date.now();
    if (left <= 0) return stopWatching();
    watchTimer = setTimeout(() => {
      collect(document.images); // catch images that loaded without an event we saw
      armWatch();
    }, left + 50);
  }
  function stopWatching() {
    document.removeEventListener("load", onImageLoad, true);
    window.removeEventListener("scroll", onScroll, true);
  }

  async function startScan() {
    scanStarted = Date.now();
    watchUntil = scanStarted + WATCH_MS;
    document.addEventListener("load", onImageLoad, true); // load does not bubble: capture
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    armWatch();
    const n = await collect(document.images);
    const total = entries.size;
    showBanner(
      n ? `DeepTrace: scanning ${n} new image${n === 1 ? "" : "s"}. Scroll within 2 s to ` +
          "include lazy-loaded images."
        : total ? "DeepTrace: no new images to scan on this page."
          : "DeepTrace: no images of at least 64×64 px found yet. Scroll within 2 s to " +
            "include lazy-loaded images.",
      { kind: "info", timeoutMs: 5000 });
  }

  // ---- results ---------------------------------------------------------------------

  function pendingCount() {
    let n = 0;
    for (const e of entries.values()) if (e.state === "pending" || e.state === "explaining") n++;
    return n;
  }

  function startPinging() {
    if (pingTimer) return;
    pingTimer = setInterval(() => {
      const now = Date.now();
      for (const [id, e] of entries) {
        if ((e.state === "pending" || e.state === "explaining") &&
            now - e.since > RESULT_TIMEOUT_MS) {
          finish(id, e.state === "explaining" ? "explain" : "predict",
            { error: "timeout", message: "no answer from the API; scan again" });
        }
      }
      if (!pendingCount()) {
        clearInterval(pingTimer);
        pingTimer = null;
        return;
      }
      chrome.runtime.sendMessage({ type: "deeptrace:ping" }).catch(() => {});
    }, PING_MS);
  }

  function finish(id, mode, body) {
    const e = entries.get(id);
    if (!e) return;
    if (mode === "explain") {
      e.state = "done";
      render(e);
      if (body && (body.error || body.status !== "ok")) {
        const why = body.message || (body.status === "no_face" ? "no face found" : body.status);
        showBanner(`DeepTrace: the heatmap could not be made: ${why}.`);
      }
      return;
    }
    e.result = body;
    e.state = "done";
    render(e);
    place(e);
  }

  function onBadgeClick(id) {
    const e = entries.get(id);
    if (!e || e.state !== "done") return;
    const v = badgeFor(e.result, showConfidence);
    if (v.kind === "error") {
      showBanner(`DeepTrace: ${v.title}`);
      return;
    }
    if (v.faceIndex == null) return; // no face: nothing to explain
    e.state = "explaining";
    e.since = Date.now();
    render(e);
    chrome.runtime
      .sendMessage({ type: "deeptrace:explain", id, url: e.sendUrl, faceIndex: v.faceIndex })
      .catch(() => finish(id, "explain", { error: "extension", message: "extension restarted" }));
    startPinging();
  }

  function apiDown(message) {
    for (const [id, e] of entries) {
      if (e.state === "pending" || e.state === "explaining") {
        finish(id, e.state === "explaining" ? "explain" : "predict",
          { error: "api_down", message: "the DeepTrace API is not running" });
      }
    }
    showBanner(message, { settings: true });
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (!msg || typeof msg.type !== "string") return;
    if (msg.type === "deeptrace:start") {
      if (msg.apiUp) startScan();
      else showBanner(msg.message, { settings: true });
    } else if (msg.type === "deeptrace:result") {
      finish(msg.id, msg.mode, msg.body);
    } else if (msg.type === "deeptrace:api-down") {
      apiDown(msg.message);
    }
  });

  // ---- settings --------------------------------------------------------------------

  chrome.storage.local.get({ showConfidence: true }).then((s) => {
    showConfidence = s.showConfidence !== false;
    for (const e of entries.values()) render(e);
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.showConfidence) {
      showConfidence = changes.showConfidence.newValue !== false;
      for (const e of entries.values()) render(e);
    }
  });
})();
