// End-to-end smoke test for the extension: a headless Chromium browser loads the unpacked
// extension, scans a local test page against a running DeepTrace API, and is driven over the
// DevTools protocol. Not part of CI (it needs a browser and the API with weights).
//
//   1. start the API:   uv run python -m deeptrace serve
//   2. run:             node scripts/extension_smoke/smoke.mjs
//
// Branded Google Chrome 137+ ignores --load-extension, so this uses Microsoft Edge on Windows
// by default; set BROWSER to Chromium or Chrome for Testing elsewhere.
import { execFileSync, spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..", "..");
const EXT = join(REPO, "extension");
const API = process.env.DEEPTRACE_API || "http://127.0.0.1:8000";
const BROWSER = process.env.BROWSER ||
  (process.platform === "win32"
    ? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    : "chromium");
const CDP_PORT = 9334;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(...a);

if (!(await fetch(`${API}/health`).then((r) => r.ok).catch(() => false))) {
  console.error(`DeepTrace API not running at ${API}: start it with uv run python -m deeptrace serve`);
  process.exit(2);
}

// ---- test site -------------------------------------------------------------------------
const work = mkdtempSync(join(tmpdir(), "deeptrace-smoke-"));
const siteDir = join(work, "site");
const makeSite = (...extra) =>
  execFileSync("uv", ["run", "python", join(HERE, "make_site.py"), siteDir, ...extra],
    { cwd: REPO, stdio: "inherit" });
makeSite();
const TYPES = { ".html": "text/html", ".jpg": "image/jpeg", ".gif": "image/gif",
  ".svg": "image/svg+xml" };
const server = createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "");
  try {
    if (name.includes("..")) throw new Error("bad path");
    const body = readFileSync(join(siteDir, name));
    res.writeHead(200, { "content-type": TYPES[extname(name)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const SITE = `http://127.0.0.1:${server.address().port}`;

// ---- browser + DevTools protocol -------------------------------------------------------
const browser = spawn(BROWSER, [
  "--headless=new", `--user-data-dir=${join(work, "profile")}`,
  `--remote-debugging-port=${CDP_PORT}`, "--no-first-run",
  `--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`,
  "--window-size=1200,900", "about:blank",
], { stdio: "ignore" });

let version;
for (let i = 0; i < 40 && !version; i++) {
  await sleep(500);
  version = await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)
    .then((r) => r.json()).catch(() => null);
}
if (!version) throw new Error(`could not start ${BROWSER}`);
const ws = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let nextId = 1;
const waiting = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && waiting.has(d.id)) {
    const { res, rej } = waiting.get(d.id);
    waiting.delete(d.id);
    if (d.error) rej(new Error(JSON.stringify(d.error)));
    else res(d.result);
  }
};
const send = (method, params = {}, sessionId) =>
  new Promise((res, rej) => {
    const id = nextId++;
    waiting.set(id, { res, rej });
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
const evaluate = async (session, expression) => {
  const r = await send("Runtime.evaluate",
    { expression, awaitPromise: true, returnByValue: true }, session);
  if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
const attach = async (targetId) =>
  (await send("Target.attachToTarget", { targetId, flatten: true })).sessionId;
const targets = async () => (await send("Target.getTargets")).targetInfos;

async function serviceWorker() {
  for (let i = 0; i < 20; i++) {
    const t = (await targets()).find((x) =>
      x.type === "service_worker" && x.url.endsWith("/background.js"));
    if (t) {
      const session = await attach(t.targetId);
      // Edge 157+ lists the worker target before background.js has finished running.
      for (let j = 0; j < 40; j++) {
        const ready = await evaluate(session, "typeof scanTab === 'function'").catch(() => false);
        if (ready) return { session, targetId: t.targetId, url: t.url };
        await sleep(250);
      }
      throw new Error("background.js never defined scanTab");
    }
    await sleep(500);
  }
  throw new Error("extension service worker not found (did the extension load?)");
}

async function openPage(url) {
  const { targetId } = await send("Target.createTarget", { url });
  const session = await attach(targetId);
  await send("Page.enable", {}, session);
  await send("DOM.enable", {}, session);
  await sleep(2500);
  return session;
}

/** Same code path as clicking the toolbar button (scanTab is the onClicked listener). */
async function scan(sw, path) {
  const tab = await evaluate(sw.session,
    `chrome.tabs.query({}).then((ts) => {
       // Edge 157+ hides tab URLs until the user grants site access to an unpacked
       // extension; then fall back to the newest tab, which openPage() just created.
       const t = ts.find((x) => (x.url || x.pendingUrl || "") === "${SITE}/${path}") ||
         (ts.every((x) => !x.url) ? ts.reduce((a, b) => (b.id > a.id ? b : a)) : undefined);
       if (!t) throw new Error("tab not found; tabs: " + JSON.stringify(ts.map((x) =>
         ({ id: x.id, url: x.url, pendingUrl: x.pendingUrl, status: x.status }))));
       return t.id;
     })`);
  await evaluate(sw.session, `scanTab({id: ${tab}})`);
}

/** Badges and banners live in a closed shadow root; the DevTools protocol can pierce it. */
async function overlay(session) {
  const { root } = await send("DOM.getDocument", { depth: -1, pierce: true }, session);
  const out = { badges: [], banners: [] };
  const text = (n) => (n.nodeType === 3 ? n.nodeValue : (n.children || []).map(text).join(""));
  const walk = (n) => {
    const attrs = n.attributes || [];
    const i = attrs.indexOf("class");
    const cls = i >= 0 && i % 2 === 0 ? attrs[i + 1] : "";
    if (/\bbadge\b/.test(cls)) out.badges.push({ nodeId: n.nodeId, text: text(n) });
    if (/\bbanner\b/.test(cls)) out.banners.push(text(n));
    for (const c of [...(n.children || []), ...(n.shadowRoots || [])]) walk(c);
  };
  walk(root);
  return out;
}

async function settle(session, maxMs = 90000) {
  const t0 = Date.now();
  let o;
  do {
    await sleep(1000);
    o = await overlay(session);
  } while (Date.now() - t0 < maxMs && o.badges.some((b) => /Scanning|Heatmap/.test(b.text)));
  return o;
}

let failures = 0;
const check = (ok, what) => {
  log(`${ok ? "PASS" : "FAIL"}  ${what}`);
  if (!ok) failures++;
};

try {
  const sw = await serviceWorker();
  log("extension loaded:", sw.url);

  // 1. filtering, dedupe, data:/blob: URLs and a lazy-loaded image
  const page = await openPage(`${SITE}/page.html`);
  await scan(sw, "page.html");
  await sleep(800);
  await evaluate(page, "window.scrollTo(0, document.body.scrollHeight)"); // lazy image loads
  await sleep(300);
  await evaluate(page, "window.scrollTo(0, 0)");
  const o = await settle(page);
  log("badges:", o.badges.map((b) => b.text).join(" | "));
  const shot = await send("Page.captureScreenshot", { format: "png" }, page);
  writeFileSync(join(work, "page.png"), Buffer.from(shot.data, "base64"));
  // scanned: aldrin, multiface, noface, armstrong (3 copies -> 1), corrupt (the browser draws
  // the readable part, the API rejects it), data: URL, blob: URL, lazy image = 8
  check(o.badges.length === 8, `8 badges after filtering and dedupe (got ${o.badges.length})`);
  check(o.badges.filter((b) => b.text === "No face").length === 1, "no-face image says No face");
  check(o.badges.filter((b) => b.text === "Error").length === 1, "only the corrupt JPEG errors");
  check(o.badges.every((b) =>
    /^(Real|Fake|Uncertain) · \d+% fake$|^No face$|^Error$/.test(b.text)), "badge text format");

  const geom = `JSON.stringify([document.body.scrollHeight, document.body.scrollWidth,
    ...[...document.images].flatMap((i) => {
      const r = i.getBoundingClientRect(); return [r.x, r.y + scrollY, r.width, r.height]; })])`;
  const withBadges = await evaluate(page, geom);
  await evaluate(page, `document.querySelector("deeptrace-overlay").style.display = "none"`);
  const without = await evaluate(page, geom);
  await evaluate(page, `document.querySelector("deeptrace-overlay").style.display = ""`);
  check(withBadges === without, "badges do not move or resize anything on the page");

  // 2. clicking a badge opens the Grad-CAM popup
  const target = o.badges.find((b) => /^(Real|Fake|Uncertain)/.test(b.text));
  const { model } = await send("DOM.getBoxModel", { nodeId: target.nodeId }, page);
  const [x1, y1, , , x3, y3] = model.border;
  for (const type of ["mousePressed", "mouseReleased"]) {
    await send("Input.dispatchMouseEvent",
      { type, x: (x1 + x3) / 2, y: (y1 + y3) / 2, button: "left", clickCount: 1 }, page);
  }
  let heat;
  for (let i = 0; i < 120 && !heat; i++) { // the first /explain loads the FP32 model
    await sleep(500);
    heat = (await targets()).find((t) => t.url.includes("/heatmap.html#"));
  }
  check(!!heat, "clicking a badge opens the heatmap popup");
  if (heat) {
    const hs = await attach(heat.targetId);
    await sleep(1000);
    const h = JSON.parse(await evaluate(hs, `JSON.stringify({
      verdict: document.getElementById("verdict").textContent,
      crop: document.getElementById("crop").naturalWidth,
      image: document.getElementById("image").naturalWidth })`));
    log("heatmap:", JSON.stringify(h));
    check(h.crop > 0 && h.image > 0 && h.verdict.length > 0, "heatmap images render under the CSP");
  }

  // 3. service worker stopped mid-scan: the queue resumes from chrome.storage.session
  makeSite("--many");
  const many = await openPage(`${SITE}/many.html`);
  await scan(sw, "many.html");
  await sleep(150);
  const saved = await evaluate(sw.session,
    `chrome.storage.session.get("queue").then((s) => (s.queue ? s.queue.items.length : 0))`);
  await send("Target.closeTarget", { targetId: sw.targetId });
  await sleep(300);
  check(!(await targets()).some((t) => t.targetId === sw.targetId),
    "service worker stopped mid-scan");
  check(saved > 0, `queue was in chrome.storage.session at shutdown (${saved} items)`);
  const o2 = await settle(many);
  check(o2.badges.length === 16 && !o2.badges.some((b) => /Scanning|Error/.test(b.text)),
    "all 16 images finished after the worker restarted");

  // 4. API unreachable: clear message (a message from an extension page wakes the worker)
  const opt = await openPage(`chrome-extension://${new URL(sw.url).host}/options.html`);
  await evaluate(opt, `chrome.runtime.sendMessage({type: "wake"}).catch(() => 0)`);
  const sw2 = await serviceWorker();
  await evaluate(sw2.session, `chrome.storage.local.set({apiUrl: "http://127.0.0.1:9"})`);
  await scan(sw2, "page.html");
  await sleep(1500);
  const o3 = await overlay(page);
  check(o3.banners.some((t) => t.startsWith("DeepTrace API not running") &&
    t.includes("uv run python -m deeptrace serve")), "API-down banner says how to start it");
  await evaluate(sw2.session, `chrome.storage.local.remove("apiUrl")`);
} catch (e) {
  failures++;
  console.error(e);
} finally {
  await send("Browser.close").catch(() => {});
  browser.kill();
  server.close();
  await sleep(500);
  if (failures) {
    log(`kept for debugging (screenshot page.png): ${work}`);
  } else {
    try {
      rmSync(work, { recursive: true, force: true });
    } catch {
      // the browser may still hold profile files on Windows
    }
  }
  log(failures ? `${failures} FAILED` : "ALL PASSED");
  process.exit(failures ? 1 : 0);
}
