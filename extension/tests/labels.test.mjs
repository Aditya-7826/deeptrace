import assert from "node:assert/strict";
import { test } from "node:test";

import "../lib/labels.js";

const { badgeFor, apiDownMessage, normalizeApiUrl, isLocalApi, DEFAULT_API_URL } =
  globalThis.DeepTraceLabels;

const face = (index, p, label) => ({ index, box: [0, 0, 100, 100], probability_fake: p, label });
const ok = (...faces) => ({ status: "ok", faces, disclaimer: "probabilistic estimate" });

test("real, fake and uncertain badges with confidence", () => {
  assert.equal(badgeFor(ok(face(0, 0.08, "real"))).text, "Real · 8% fake");
  assert.equal(badgeFor(ok(face(0, 0.87, "fake"))).text, "Fake · 87% fake");
  assert.equal(badgeFor(ok(face(0, 0.55, "uncertain"))).text, "Uncertain · 55% fake");
  const v = badgeFor(ok(face(0, 0.87, "fake")));
  assert.equal(v.kind, "fake");
  assert.equal(v.faceIndex, 0);
  assert.match(v.title, /Probability fake: 87%/);
  assert.match(v.title, /probabilistic estimate/);
});

test("the number is P(fake) even when a low threshold labels 36% as fake", () => {
  assert.equal(badgeFor(ok(face(0, 0.36, "fake"))).text, "Fake · 36% fake");
});

test("the confidence toggle hides the number", () => {
  assert.equal(badgeFor(ok(face(0, 0.87, "fake")), false).text, "Fake");
});

test("several faces: the most likely fake is shown and explained", () => {
  const v = badgeFor(ok(face(0, 0.1, "real"), face(1, 0.9, "fake"), face(2, 0.5, "uncertain")));
  assert.equal(v.text, "Fake · 90% fake (3 faces)");
  assert.equal(v.faceIndex, 1);
  assert.equal(v.faces, 3);
});

test("no face and face too small", () => {
  assert.deepEqual(
    [badgeFor({ status: "no_face" }).text, badgeFor({ status: "face_too_small" }).text],
    ["No face", "No face"]);
  assert.equal(badgeFor({ status: "no_face" }).faceIndex, null);
  assert.match(badgeFor({ status: "face_too_small" }).title, /too small/);
});

test("errors keep the API message", () => {
  const v = badgeFor({ error: "unsupported_type", message: "file signature and content disagree" });
  assert.equal(v.kind, "error");
  assert.equal(v.text, "Error");
  assert.equal(v.title, "file signature and content disagree");
  assert.equal(badgeFor(null).kind, "error");
  assert.equal(badgeFor({ status: "ok", faces: [] }).kind, "error");
});

test("the API-down message says how to start it", () => {
  const m = apiDownMessage("http://127.0.0.1:8000");
  assert.match(m, /^DeepTrace API not running/);
  assert.match(m, /uv run python -m deeptrace serve/);
  assert.match(m, /127\.0\.0\.1:8000/);
});

test("settings URL validation", () => {
  assert.equal(DEFAULT_API_URL, "http://127.0.0.1:8000");
  assert.equal(normalizeApiUrl(" http://127.0.0.1:8000/ "), "http://127.0.0.1:8000");
  assert.equal(normalizeApiUrl("https://api.example.test/deeptrace/"),
    "https://api.example.test/deeptrace");
  assert.equal(normalizeApiUrl("ftp://x.test"), null);
  assert.equal(normalizeApiUrl("javascript:alert(1)"), null);
  assert.equal(normalizeApiUrl("http://user:pw@x.test"), null);
  assert.equal(normalizeApiUrl("http://x.test/?a=1"), null);
  assert.equal(normalizeApiUrl("not a url"), null);
  assert.equal(isLocalApi("http://localhost:8000"), true);
  assert.equal(isLocalApi("http://127.0.0.1:9000"), true);
  assert.equal(isLocalApi("https://api.example.test"), false);
});
