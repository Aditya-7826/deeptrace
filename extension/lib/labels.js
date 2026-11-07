// Turns API responses into badge text. Pure functions, shared by the content script and tests.
(() => {
  "use strict";

  const DEFAULT_API_URL = "http://127.0.0.1:8000";
  const START_COMMAND = "uv run python -m deeptrace serve";

  const pct = (p) => `${Math.round(p * 100)}%`;

  /**
   * Badge for one image. Several faces: the face most likely to be fake is shown.
   * @param {object} result  /predict JSON, or {error, message} from the API or the extension
   * @param {boolean} showConfidence
   * @returns {{kind: string, text: string, title: string, faceIndex: number | null,
   *            faces: number}}
   */
  function badgeFor(result, showConfidence = true) {
    if (!result || result.error) {
      const message = (result && result.message) || "unknown error";
      return { kind: "error", text: "Error", title: message, faceIndex: null, faces: 0 };
    }
    if (result.status === "no_face" || result.status === "face_too_small") {
      const title =
        result.status === "no_face" ? "No face detected" : "Face too small to analyse";
      return { kind: "no_face", text: "No face", title, faceIndex: null, faces: 0 };
    }
    const faces = Array.isArray(result.faces) ? result.faces : [];
    if (result.status !== "ok" || !faces.length) {
      return { kind: "error", text: "Error", title: "unexpected API response", faceIndex: null,
        faces: 0 };
    }
    const top = faces.reduce((a, b) => (b.probability_fake > a.probability_fake ? b : a));
    const p = top.probability_fake;
    const name = { real: "Real", fake: "Fake", uncertain: "Uncertain" }[top.label] || "Error";
    const kind = name === "Error" ? "error" : top.label;
    let text = name;
    // Always the calibrated probability of "fake", so the number never changes meaning
    // with the label (the decision threshold need not be 50%).
    if (showConfidence && kind !== "error") text += ` · ${pct(p)} fake`;
    let title = `Probability fake: ${pct(p)}`;
    if (faces.length > 1) {
      text += ` (${faces.length} faces)`;
      title += ` (face ${top.index + 1} of ${faces.length}, most likely fake shown)`;
    }
    if (result.disclaimer) title += `\n${result.disclaimer}`;
    title += "\nClick for the Grad-CAM heatmap.";
    return { kind, text, title, faceIndex: top.index, faces: faces.length };
  }

  function apiDownMessage(apiUrl = DEFAULT_API_URL) {
    return (
      `DeepTrace API not running at ${apiUrl}: start it with ` +
      `\`${START_COMMAND}\` in the deeptrace folder, then scan again.`
    );
  }

  /** Validates the settings-page API URL; returns the normalized origin or null. */
  function normalizeApiUrl(text) {
    try {
      const u = new URL(String(text).trim());
      if (u.protocol !== "http:" && u.protocol !== "https:") return null;
      if (u.username || u.password || u.search || u.hash) return null;
      return u.href.replace(/\/+$/, "");
    } catch {
      return null;
    }
  }

  function isLocalApi(apiUrl) {
    try {
      const host = new URL(apiUrl).hostname;
      return host === "127.0.0.1" || host === "localhost" || host === "[::1]";
    } catch {
      return false;
    }
  }

  globalThis.DeepTraceLabels = {
    DEFAULT_API_URL, START_COMMAND, badgeFor, apiDownMessage, normalizeApiUrl, isLocalApi,
  };
})();
