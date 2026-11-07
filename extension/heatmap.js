"use strict";

(async () => {
  const key = decodeURIComponent(location.hash.slice(1));
  const $ = (id) => document.getElementById(id);
  if (!key.startsWith("heatmap:")) {
    $("missing").hidden = false;
    return;
  }
  const data = (await chrome.storage.session.get(key))[key];
  if (!data) {
    $("missing").hidden = false;
    return;
  }
  // The overlays live in this window now; free the session storage they used.
  chrome.storage.session.remove(key);

  $("crop").src = `data:image/png;base64,${data.crop}`;
  $("image").src = `data:image/png;base64,${data.image}`;
  $("views").hidden = false;
  if (data.face) {
    const p = data.face.probability_fake;
    const name = { real: "Real", fake: "Fake", uncertain: "Uncertain" }[data.face.label] || "?";
    $("verdict").textContent =
      `${name}: probability fake ${Math.round(p * 100)}% (face ${data.faceIndex + 1})`;
    $("verdict").classList.add(data.face.label);
  }
  if (data.target) $("target").textContent = `Heatmap shows ${data.target}.`;
  $("disclaimer").textContent = data.disclaimer ? `Note: ${data.disclaimer}.` : "";
  const model = data.modelVersion ? ` · model ${data.modelVersion}` : "";
  $("source").textContent = `Source: ${data.source}${model}`;
})();
