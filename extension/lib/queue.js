// Request queue for the service worker: at most N requests in flight, backoff on HTTP 429,
// and a JSON snapshot so the queue survives an MV3 service-worker shutdown.
// Pure logic (no chrome.* calls); the clock is injectable for tests.
(() => {
  "use strict";

  /** Retry-After header (seconds or HTTP date) to milliseconds, or null. */
  function parseRetryAfter(value, now = Date.now()) {
    if (value == null || value === "") return null;
    const text = String(value).trim();
    if (/^\d+(\.\d+)?$/.test(text)) return Math.round(Number(text) * 1000);
    const at = Date.parse(text);
    return Number.isNaN(at) ? null : Math.max(0, at - now);
  }

  /** Delay before retry `attempt` (1-based): Retry-After if given, else exponential. */
  function backoffDelay(attempt, retryAfterMs, baseMs = 1000, maxMs = 30000) {
    if (retryAfterMs != null) return Math.min(Math.max(retryAfterMs, 0), maxMs);
    return Math.min(baseMs * 2 ** Math.max(attempt - 1, 0), maxMs);
  }

  class ScanQueue {
    /**
     * @param {{maxInFlight?: number, maxAttempts?: number, baseDelayMs?: number,
     *          maxDelayMs?: number, now?: () => number}} opts
     */
    constructor(opts = {}) {
      this.maxInFlight = opts.maxInFlight ?? 2;
      this.maxAttempts = opts.maxAttempts ?? 5;
      this.baseDelayMs = opts.baseDelayMs ?? 1000;
      this.maxDelayMs = opts.maxDelayMs ?? 30000;
      this.now = opts.now ?? Date.now;
      this.pending = []; // FIFO of {key, attempts, notBefore, ...payload}
      this.inFlight = new Map(); // key -> item
    }

    has(key) {
      return this.inFlight.has(key) || this.pending.some((it) => it.key === key);
    }

    /**
     * Add an item ({key, ...payload}); returns false if the key is already queued.
     * `front` puts it ahead of waiting items (used for a user-clicked heatmap).
     */
    add(item, { front = false } = {}) {
      if (!item || item.key == null || this.has(item.key)) return false;
      const entry = { attempts: 0, notBefore: 0, ...item };
      if (front) this.pending.unshift(entry);
      else this.pending.push(entry);
      return true;
    }

    /** Next item that may start now (marked in flight), or null. */
    next() {
      if (this.inFlight.size >= this.maxInFlight) return null;
      const t = this.now();
      const i = this.pending.findIndex((it) => it.notBefore <= t);
      if (i < 0) return null;
      const [item] = this.pending.splice(i, 1);
      item.attempts += 1;
      this.inFlight.set(item.key, item);
      return item;
    }

    /** Start as many items as the in-flight limit allows. */
    drain() {
      const started = [];
      for (let it = this.next(); it; it = this.next()) started.push(it);
      return started;
    }

    /** Finished (success or permanent failure): forget the item. */
    done(key) {
      return this.inFlight.delete(key);
    }

    /**
     * Rate limited or transient error: requeue with backoff.
     * @returns {{requeued: true, delayMs: number} | {requeued: false}}
     */
    retry(key, retryAfterMs = null) {
      const item = this.inFlight.get(key);
      if (!item) return { requeued: false };
      this.inFlight.delete(key);
      if (item.attempts >= this.maxAttempts) return { requeued: false };
      const delayMs = backoffDelay(item.attempts, retryAfterMs, this.baseDelayMs, this.maxDelayMs);
      item.notBefore = this.now() + delayMs;
      this.pending.push(item);
      return { requeued: true, delayMs };
    }

    /** Drop queued and in-flight items matching `pred` (e.g. a closed tab). */
    removeWhere(pred) {
      const before = this.size;
      this.pending = this.pending.filter((it) => !pred(it));
      for (const [key, it] of this.inFlight) if (pred(it)) this.inFlight.delete(key);
      return before - this.size;
    }

    /** Milliseconds until the next pending item becomes ready (0 = now), or null if none. */
    msUntilReady() {
      if (!this.pending.length) return null;
      const t = this.now();
      return Math.max(0, Math.min(...this.pending.map((it) => it.notBefore)) - t);
    }

    get size() {
      return this.pending.length + this.inFlight.size;
    }

    get inFlightCount() {
      return this.inFlight.size;
    }

    /** Snapshot for chrome.storage.session. In-flight work is saved as pending. */
    toJSON() {
      return {
        version: 1,
        items: [...this.inFlight.values(), ...this.pending].map((it) => ({ ...it })),
      };
    }

    /** Rebuild after a restart. Interrupted requests go back to the front of the queue. */
    static fromJSON(state, opts = {}) {
      const q = new ScanQueue(opts);
      if (state && state.version === 1 && Array.isArray(state.items)) {
        for (const it of state.items) q.add(it);
      }
      return q;
    }
  }

  globalThis.DeepTraceQueue = { ScanQueue, backoffDelay, parseRetryAfter };
})();
