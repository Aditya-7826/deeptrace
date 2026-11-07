import assert from "node:assert/strict";
import { test } from "node:test";

import "../lib/filter.js";

const { MIN_SIDE, skipReason, normalizeUrl, isSvg, UrlDeduper } = globalThis.DeepTraceFilter;

const img = (over = {}) => ({
  src: "https://example.test/photo.jpg",
  naturalWidth: 800,
  naturalHeight: 600,
  width: 400,
  height: 300,
  ...over,
});

test("a normal photo is scanned", () => {
  assert.equal(skipReason(img()), null);
});

test("images under 64x64 are skipped (natural or rendered size)", () => {
  assert.equal(MIN_SIDE, 64);
  assert.equal(skipReason(img({ naturalWidth: 63 })), "too_small");
  assert.equal(skipReason(img({ naturalHeight: 63 })), "too_small");
  assert.equal(skipReason(img({ width: 40, height: 40 })), "too_small", "thumbnail of big file");
  assert.equal(skipReason(img({ width: 63.4 })), "too_small");
  assert.equal(skipReason(img({ naturalWidth: 64, naturalHeight: 64, width: 64, height: 64 })),
    null, "exactly 64 px is allowed");
  assert.equal(skipReason(img({ width: 63.6, height: 64 })), null, "rounded CSS size");
});

test("tracking pixels are skipped", () => {
  assert.equal(skipReason(img({ naturalWidth: 1, naturalHeight: 1 })), "tracking_pixel");
  assert.equal(skipReason(img({ naturalWidth: 2, naturalHeight: 2 })), "tracking_pixel");
  assert.equal(skipReason(img({ width: 1, height: 1 })), "tracking_pixel", "big file shown at 1px");
  assert.equal(skipReason(img({ width: 0, height: 0 })), "tracking_pixel");
});

test("SVG is skipped (path, query string, data URL)", () => {
  assert.equal(skipReason(img({ src: "https://x.test/logo.svg" })), "svg");
  assert.equal(skipReason(img({ src: "https://x.test/logo.SVG?v=3#top" })), "svg");
  assert.equal(skipReason(img({ src: "https://x.test/a.svgz" })), "svg");
  assert.equal(skipReason(img({ src: "data:image/svg+xml;base64,PHN2Zz4=" })), "svg");
  assert.equal(isSvg("https://x.test/svg/photo.jpg"), false, "folder named svg");
  assert.equal(isSvg("https://x.test/photo.jpg?format=svg"), false);
});

test("only http, https, data and blob URLs are allowed", () => {
  assert.equal(skipReason(img({ src: "" })), "no_src");
  assert.equal(skipReason(null), "no_src");
  assert.equal(skipReason(img({ src: "chrome-extension://abc/x.png" })), "unsupported_scheme");
  assert.equal(skipReason(img({ src: "file:///C:/x.jpg" })), "unsupported_scheme");
  assert.equal(skipReason(img({ src: "javascript:alert(1)" })), "unsupported_scheme");
  assert.equal(skipReason(img({ src: "data:image/png;base64,iVBORw0KGgo=" })), null);
  assert.equal(skipReason(img({ src: "blob:https://example.test/1234-5678" })), null);
  assert.equal(skipReason(img({ src: "HTTPS://EXAMPLE.TEST/A.JPG" })), null);
});

test("normalizeUrl resolves relative URLs and drops the fragment", () => {
  const base = "https://example.test/gallery/page.html";
  assert.equal(normalizeUrl("../img/a.jpg#zoom", base), "https://example.test/img/a.jpg");
  assert.equal(normalizeUrl("a.jpg?w=200", base), "https://example.test/gallery/a.jpg?w=200");
  assert.equal(normalizeUrl("data:image/png;base64,AAA#x", base), "data:image/png;base64,AAA#x");
  assert.equal(normalizeUrl("blob:https://example.test/1", base), "blob:https://example.test/1");
  assert.equal(normalizeUrl("http://[bad", base), null);
});

test("dedupe: the same URL is queued once", () => {
  const d = new UrlDeduper();
  const base = "https://example.test/";
  assert.equal(d.add(normalizeUrl("/a.jpg", base)), true);
  assert.equal(d.add(normalizeUrl("https://example.test/a.jpg", base)), false);
  assert.equal(d.add(normalizeUrl("/a.jpg#2", base)), false, "fragment-only difference");
  assert.equal(d.add(normalizeUrl("/a.jpg?w=2", base)), true, "different query = different file");
  assert.equal(d.add(null), false, "unparseable URL");
  assert.equal(d.size, 2);
});
