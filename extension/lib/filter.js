// Which page images are worth sending to the API. Pure functions, no DOM access.
// Loaded as a classic script (content script, service worker) and by Node tests,
// so it attaches to globalThis instead of using `export`.
(() => {
  "use strict";

  const MIN_SIDE = 64; // the API rejects faces smaller than this anyway
  const PIXEL_MAX_SIDE = 2; // 1x1 / 2x2 images are tracking pixels
  const SCHEMES = new Set(["http:", "https:", "data:", "blob:"]);

  function scheme(url) {
    const m = /^([a-z][a-z0-9+.-]*:)/i.exec(url || "");
    return m ? m[1].toLowerCase() : "";
  }

  function isSvg(url) {
    const s = scheme(url);
    if (s === "data:") return /^data:image\/svg\+xml/i.test(url);
    if (s === "blob:") return false; // type unknown until fetched; the API rejects SVG
    let path = url;
    try {
      path = new URL(url).pathname;
    } catch {
      // keep the raw string
    }
    return /\.svgz?$/i.test(path);
  }

  /**
   * Why an image should be skipped, or null if it should be scanned.
   * @param {{src: string, naturalWidth: number, naturalHeight: number,
   *          width: number, height: number}} d  natural size and rendered (CSS px) size
   * @returns {null | "no_src" | "unsupported_scheme" | "svg" | "tracking_pixel" | "too_small"}
   */
  function skipReason(d) {
    if (!d || !d.src) return "no_src";
    if (!SCHEMES.has(scheme(d.src))) return "unsupported_scheme";
    if (isSvg(d.src)) return "svg";
    const nw = d.naturalWidth | 0;
    const nh = d.naturalHeight | 0;
    const rw = Math.round(d.width || 0);
    const rh = Math.round(d.height || 0);
    if (Math.max(nw, nh) <= PIXEL_MAX_SIDE || Math.max(rw, rh) <= PIXEL_MAX_SIDE) {
      return "tracking_pixel";
    }
    if (Math.min(nw, nh) < MIN_SIDE || Math.min(rw, rh) < MIN_SIDE) return "too_small";
    return null;
  }

  /** Absolute URL without the #fragment (same image, different anchor = duplicate). */
  function normalizeUrl(src, base) {
    if (scheme(src) === "data:" || scheme(src) === "blob:") return src;
    try {
      const u = new URL(src, base);
      u.hash = "";
      return u.href;
    } catch {
      return null;
    }
  }

  /** Remembers URLs already queued in this scan session. */
  class UrlDeduper {
    constructor() {
      this.seen = new Set();
    }

    /** True the first time a URL is offered, false for repeats. */
    add(url) {
      if (url == null || this.seen.has(url)) return false;
      this.seen.add(url);
      return true;
    }

    get size() {
      return this.seen.size;
    }
  }

  globalThis.DeepTraceFilter = { MIN_SIDE, skipReason, normalizeUrl, isSvg, UrlDeduper };
})();
