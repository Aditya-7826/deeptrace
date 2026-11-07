"use strict";

const { DEFAULT_API_URL, normalizeApiUrl, isLocalApi, apiDownMessage } = globalThis.DeepTraceLabels;
const DEFAULTS = { apiUrl: DEFAULT_API_URL, showConfidence: true };
const $ = (id) => document.getElementById(id);

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}

function updateWarning() {
  const url = normalizeApiUrl($("apiUrl").value);
  $("remoteWarn").hidden = !url || isLocalApi(url);
}

async function load() {
  const s = await chrome.storage.local.get(DEFAULTS);
  $("apiUrl").value = s.apiUrl;
  $("showConfidence").checked = s.showConfidence !== false;
  updateWarning();
}

function readForm() {
  const apiUrl = normalizeApiUrl($("apiUrl").value);
  if (!apiUrl) {
    setStatus("Enter an http:// or https:// URL, for example http://127.0.0.1:8000", "bad");
    return null;
  }
  return { apiUrl, showConfidence: $("showConfidence").checked };
}

$("apiUrl").addEventListener("input", updateWarning);

$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const s = readForm();
  if (!s) return;
  await chrome.storage.local.set(s);
  $("apiUrl").value = s.apiUrl;
  setStatus("Saved.", "ok");
});

$("reset").addEventListener("click", async () => {
  await chrome.storage.local.set(DEFAULTS);
  await load();
  setStatus("Defaults restored.", "ok");
});

$("test").addEventListener("click", async () => {
  const s = readForm();
  if (!s) return;
  setStatus("Connecting…");
  try {
    const res = await fetch(`${s.apiUrl}/version`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const v = await res.json();
    setStatus(`Connected: model ${v.model_name} (${v.model_version}), ` +
      `${v.calibrated ? "calibrated" : "not calibrated"}.`, "ok");
  } catch {
    setStatus(apiDownMessage(s.apiUrl).replaceAll("`", ""), "bad");
  }
});

load();
