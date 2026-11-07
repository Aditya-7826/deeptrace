// DeepTrace service worker. It fetches image bytes (no tainted-canvas or CORS problems in
// the page), sends them to the local API with at most 2 requests in flight, and persists
// the queue in chrome.storage.session so a service-worker shutdown mid-scan loses nothing.
"use strict";

importScripts("lib/queue.js", "lib/labels.js");

const { ScanQueue, parseRetryAfter } = globalThis.DeepTraceQueue;
const { DEFAULT_API_URL, apiDownMessage } = globalThis.DeepTraceLabels;

const STATE_KEY = "queue";
const MAX_BYTES = 10 * 1024 * 1024; // the API's upload limit
const IMAGE_TIMEOUT_MS = 15000;
const API_TIMEOUT_MS = 60000;
const HEALTH_TIMEOUT_MS = 3000;
const CONTENT_FILES = ["lib/filter.js", "lib/labels.js", "content.js"];

class ExtError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

// ---- state ---------------------------------------------------------------------------

let queue = null;
let loading = null;
let saving = Promise.resolve();
let wakeTimer = null;

function loadQueue() {
  if (!loading) {
    loading = chrome.storage.session
      .get(STATE_KEY)
      .then((s) => {
        queue = ScanQueue.fromJSON(s[STATE_KEY]);
      })
      .catch(() => {
        queue = new ScanQueue();
      });
  }
  return loading;
}

function persist() {
  const snapshot = queue.toJSON();
  saving = saving
    .then(() => chrome.storage.session.set({ [STATE_KEY]: snapshot }))
    .catch((e) => console.warn("DeepTrace: could not save the queue", e));
  return saving;
}

async function getSettings() {
  return chrome.storage.local.get({ apiUrl: DEFAULT_API_URL, showConfidence: true });
}

// ---- network -------------------------------------------------------------------------

/** Image bytes from an http(s) or data: URL (blob: URLs arrive as data: URLs). */
async function fetchImage(url) {
  let res;
  try {
    res = await fetch(url, {
      credentials: "omit",
      cache: "force-cache",
      signal: AbortSignal.timeout(IMAGE_TIMEOUT_MS),
    });
  } catch {
    throw new ExtError("fetch_failed", "the image could not be downloaded");
  }
  if (!res.ok) {
    throw new ExtError("fetch_failed", `the image could not be downloaded (HTTP ${res.status})`);
  }
  const declared = Number(res.headers.get("content-length"));
  if (declared > MAX_BYTES) throw new ExtError("file_too_large", "the image is larger than 10 MB");
  const blob = await res.blob();
  if (blob.size > MAX_BYTES) throw new ExtError("file_too_large", "the image is larger than 10 MB");
  if (!blob.size) throw new ExtError("empty_file", "the image is empty");
  return blob;
}

/**
 * One API call. Returns {kind: "done", body} | {kind: "retry", retryAfterMs} |
 * {kind: "api_down"}.
 */
async function callApi(apiUrl, item, blob) {
  const form = new FormData();
  form.append("file", blob, "image");
  const path =
    item.mode === "explain" ? `/explain?face_index=${encodeURIComponent(item.faceIndex)}` : "/predict";
  let res;
  try {
    res = await fetch(apiUrl + path, {
      method: "POST",
      body: form,
      credentials: "omit",
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });
  } catch (e) {
    if (e && e.name === "TimeoutError") return { kind: "retry", retryAfterMs: null };
    return { kind: "api_down" };
  }
  if (res.status === 429 || res.status === 503 || res.status === 504) {
    return { kind: "retry", retryAfterMs: parseRetryAfter(res.headers.get("retry-after")) };
  }
  let body;
  try {
    body = await res.json();
  } catch {
    body = { error: "bad_response", message: `the API answered HTTP ${res.status} without JSON` };
  }
  return { kind: "done", body };
}

async function apiHealthy(apiUrl) {
  try {
    const res = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
    return res.ok;
  } catch {
    return false;
  }
}

// ---- delivery ------------------------------------------------------------------------

async function sendToTab(tabId, message) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
    return true;
  } catch {
    // tab closed or navigated away: its remaining work is pointless
    if (queue.removeWhere((it) => it.tabId === tabId)) await persist();
    return false;
  }
}

async function showHeatmap(item, body) {
  const key = `heatmap:${crypto.randomUUID()}`;
  await chrome.storage.session.set({
    [key]: {
      crop: body.crop_overlay_png,
      image: body.image_overlay_png,
      face: body.face || null,
      faceIndex: item.faceIndex,
      target: body.heatmap_target || "",
      source: item.url.startsWith("data:") ? "(embedded image)" : item.url,
      disclaimer: body.disclaimer || "",
      modelVersion: body.model_version || "",
    },
  });
  await chrome.windows.create({
    url: chrome.runtime.getURL(`heatmap.html#${encodeURIComponent(key)}`),
    type: "popup",
    width: 560,
    height: 720,
  });
}

async function updateBadge(tabId) {
  const left = queue.pending.filter((it) => it.tabId === tabId).length +
    [...queue.inFlight.values()].filter((it) => it.tabId === tabId).length;
  try {
    await chrome.action.setBadgeBackgroundColor({ tabId, color: "#3b5bdb" });
    await chrome.action.setBadgeText({ tabId, text: left ? String(Math.min(left, 999)) : "" });
  } catch {
    // tab is gone
  }
}

// ---- queue pump ----------------------------------------------------------------------

async function apiDown(apiUrl) {
  const tabs = new Set([...queue.pending, ...queue.inFlight.values()].map((it) => it.tabId));
  queue.removeWhere(() => true);
  await persist();
  for (const tabId of tabs) {
    await sendToTab(tabId, { type: "deeptrace:api-down", message: apiDownMessage(apiUrl) });
    await updateBadge(tabId);
  }
}

async function run(item) {
  const { apiUrl } = await getSettings();
  let outcome;
  try {
    const blob = await fetchImage(item.url);
    outcome = await callApi(apiUrl, item, blob);
  } catch (e) {
    outcome = { kind: "done", body: { error: e.code || "fetch_failed", message: e.message } };
  }

  if (outcome.kind === "api_down") {
    queue.done(item.key);
    await apiDown(apiUrl);
    return;
  }
  if (outcome.kind === "retry") {
    const r = queue.retry(item.key, outcome.retryAfterMs);
    if (!r.requeued) {
      const body = { error: "rate_limited", message: "the API stayed busy; scan again later" };
      await sendToTab(item.tabId, { type: "deeptrace:result", id: item.id, mode: item.mode, body });
    }
  } else {
    queue.done(item.key);
    const ok = !outcome.body.error && outcome.body.status === "ok";
    if (item.mode === "explain" && ok) {
      await showHeatmap(item, outcome.body);
      await sendToTab(item.tabId, { type: "deeptrace:result", id: item.id, mode: "explain",
        body: { status: "ok" } });
    } else {
      await sendToTab(item.tabId, { type: "deeptrace:result", id: item.id, mode: item.mode,
        body: outcome.body });
    }
  }
  await persist();
  await updateBadge(item.tabId);
  pump();
}

async function pump() {
  await loadQueue();
  clearTimeout(wakeTimer);
  const started = queue.drain();
  if (started.length) await persist();
  for (const item of started) {
    run(item).catch((e) => {
      console.error("DeepTrace: request failed", e);
      queue.done(item.key);
      persist();
    });
  }
  const wait = queue.msUntilReady();
  if (wait != null && queue.inFlightCount < queue.maxInFlight) {
    wakeTimer = setTimeout(pump, wait + 20);
  }
}

// ---- events (registered synchronously at top level, as MV3 requires) -----------------

/** Toolbar click: inject the content script and start a scan if the API is up. */
async function scanTab(tab) {
  if (tab.id == null) return;
  const { apiUrl } = await getSettings();
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: CONTENT_FILES });
  } catch {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "!" });
    await chrome.action.setTitle({ tabId: tab.id,
      title: "DeepTrace cannot scan this page (browser pages and the Web Store are off limits)" });
    return;
  }
  const up = await apiHealthy(apiUrl);
  await sendToTab(tab.id, {
    type: "deeptrace:start",
    apiUp: up,
    message: up ? null : apiDownMessage(apiUrl),
  });
}

chrome.action.onClicked.addListener(scanTab);

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tabId = sender.tab && sender.tab.id;
  if (!msg || typeof msg.type !== "string") return false;
  if (msg.type === "deeptrace:open-options") {
    chrome.runtime.openOptionsPage();
    return false;
  }
  if (tabId == null) return false;

  (async () => {
    await loadQueue();
    if (msg.type === "deeptrace:enqueue" && Array.isArray(msg.items)) {
      for (const it of msg.items) {
        queue.add({ key: `${tabId}:${it.id}:predict`, tabId, id: it.id, url: it.url,
          mode: "predict" });
      }
    } else if (msg.type === "deeptrace:explain") {
      queue.add({ key: `${tabId}:${msg.id}:explain`, tabId, id: msg.id, url: msg.url,
        mode: "explain", faceIndex: msg.faceIndex | 0 }, { front: true });
    }
    // "deeptrace:ping" only wakes the worker; a restarted worker resumes from storage
    await persist();
    await updateBadge(tabId);
    pump();
    const mine = queue.pending.filter((it) => it.tabId === tabId).length +
      [...queue.inFlight.values()].filter((it) => it.tabId === tabId).length;
    sendResponse({ queued: mine });
  })();
  return true; // async sendResponse
});

chrome.tabs.onRemoved.addListener(async (tabId) => {
  await loadQueue();
  if (queue.removeWhere((it) => it.tabId === tabId)) await persist();
});

// Resume work left over from a previous service-worker instance.
pump();
