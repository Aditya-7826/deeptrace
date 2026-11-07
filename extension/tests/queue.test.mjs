import assert from "node:assert/strict";
import { test } from "node:test";

import "../lib/queue.js";

const { ScanQueue, backoffDelay, parseRetryAfter } = globalThis.DeepTraceQueue;

function clock(start = 1000) {
  const c = { t: start, now: () => c.t };
  return c;
}

const item = (key, extra = {}) => ({ key, tabId: 1, url: `https://x.test/${key}.jpg`, ...extra });

test("at most two requests are in flight", () => {
  const q = new ScanQueue();
  for (const k of ["a", "b", "c", "d"]) q.add(item(k));
  const started = q.drain();
  assert.deepEqual(started.map((i) => i.key), ["a", "b"]);
  assert.equal(q.inFlightCount, 2);
  assert.equal(q.next(), null);
  q.done("a");
  assert.equal(q.next().key, "c");
  assert.equal(q.next(), null);
});

test("the in-flight limit is configurable", () => {
  const q = new ScanQueue({ maxInFlight: 1 });
  q.add(item("a"));
  q.add(item("b"));
  assert.equal(q.drain().length, 1);
});

test("items are processed in FIFO order", () => {
  const q = new ScanQueue({ maxInFlight: 10 });
  for (const k of ["a", "b", "c"]) q.add(item(k));
  assert.deepEqual(q.drain().map((i) => i.key), ["a", "b", "c"]);
});

test("duplicate keys are rejected, pending or in flight", () => {
  const q = new ScanQueue();
  assert.equal(q.add(item("a")), true);
  assert.equal(q.add(item("a")), false);
  q.next();
  assert.equal(q.add(item("a")), false);
  assert.equal(q.add({ url: "no key" }), false);
  assert.equal(q.size, 1);
});

test("front puts a clicked heatmap ahead of waiting scans", () => {
  const q = new ScanQueue({ maxInFlight: 1 });
  q.add(item("a"));
  q.add(item("b"));
  q.add(item("heat", { mode: "explain" }), { front: true });
  assert.equal(q.next().key, "heat");
});

test("429 requeues with exponential backoff", () => {
  const c = clock();
  const q = new ScanQueue({ now: c.now, baseDelayMs: 1000 });
  q.add(item("a"));
  q.next();
  assert.deepEqual(q.retry("a"), { requeued: true, delayMs: 1000 });
  assert.equal(q.next(), null, "not ready before the delay");
  assert.equal(q.msUntilReady(), 1000);
  c.t += 1000;
  assert.equal(q.next().key, "a");
  assert.deepEqual(q.retry("a"), { requeued: true, delayMs: 2000 });
  c.t += 2000;
  q.next();
  assert.deepEqual(q.retry("a"), { requeued: true, delayMs: 4000 });
});

test("Retry-After from the API wins over exponential backoff", () => {
  const c = clock();
  const q = new ScanQueue({ now: c.now });
  q.add(item("a"));
  q.next();
  assert.deepEqual(q.retry("a", 7000), { requeued: true, delayMs: 7000 });
  c.t += 6999;
  assert.equal(q.next(), null);
  c.t += 1;
  assert.equal(q.next().key, "a");
});

test("a backed-off item does not block ready items behind it", () => {
  const c = clock();
  const q = new ScanQueue({ now: c.now, maxInFlight: 1 });
  q.add(item("a"));
  q.next();
  q.retry("a", 5000);
  q.add(item("b"));
  assert.equal(q.next().key, "b");
});

test("retries stop after maxAttempts", () => {
  const c = clock();
  const q = new ScanQueue({ now: c.now, maxAttempts: 3, maxDelayMs: 10 });
  q.add(item("a"));
  for (let i = 0; i < 2; i++) {
    c.t += 100;
    q.next();
    assert.equal(q.retry("a").requeued, true);
  }
  c.t += 100;
  q.next();
  assert.deepEqual(q.retry("a"), { requeued: false });
  assert.equal(q.size, 0);
});

test("retry of an unknown key is a no-op", () => {
  const q = new ScanQueue();
  assert.deepEqual(q.retry("ghost"), { requeued: false });
});

test("backoffDelay caps at maxMs and clamps negative Retry-After", () => {
  assert.equal(backoffDelay(1, null, 1000, 30000), 1000);
  assert.equal(backoffDelay(4, null, 1000, 30000), 8000);
  assert.equal(backoffDelay(20, null, 1000, 30000), 30000);
  assert.equal(backoffDelay(1, 120000, 1000, 30000), 30000);
  assert.equal(backoffDelay(1, -5, 1000, 30000), 0);
});

test("parseRetryAfter reads seconds and HTTP dates", () => {
  const now = Date.parse("2026-11-07T12:00:00Z");
  assert.equal(parseRetryAfter("3", now), 3000);
  assert.equal(parseRetryAfter("1.5", now), 1500);
  assert.equal(parseRetryAfter("Sat, 07 Nov 2026 12:00:10 GMT", now), 10000);
  assert.equal(parseRetryAfter("Sat, 07 Nov 2026 11:00:00 GMT", now), 0);
  assert.equal(parseRetryAfter(null, now), null);
  assert.equal(parseRetryAfter("", now), null);
  assert.equal(parseRetryAfter("soon", now), null);
});

test("removeWhere drops a closed tab's pending and in-flight work", () => {
  const q = new ScanQueue();
  q.add(item("a", { tabId: 1 }));
  q.add(item("b", { tabId: 2 }));
  q.add(item("c", { tabId: 1 }));
  q.next(); // a in flight
  assert.equal(q.removeWhere((it) => it.tabId === 1), 2);
  assert.deepEqual(q.drain().map((i) => i.key), ["b"]);
});

test("snapshot survives a service-worker restart", () => {
  const c = clock();
  const q = new ScanQueue({ now: c.now });
  for (const k of ["a", "b", "c"]) q.add(item(k));
  q.next(); // a in flight when the worker dies
  q.next();
  q.retry("b", 3000); // b waiting for backoff
  const saved = JSON.parse(JSON.stringify(q.toJSON())); // what chrome.storage.session stores

  const r = ScanQueue.fromJSON(saved, { now: c.now });
  assert.equal(r.size, 3);
  assert.equal(r.inFlightCount, 0, "interrupted requests are pending again");
  const started = r.drain();
  assert.deepEqual(started.map((i) => i.key), ["a", "c"], "a resumes first; b still backing off");
  assert.equal(started[0].url, "https://x.test/a.jpg", "payload is kept");
  c.t += 3000;
  r.done("a");
  assert.equal(r.next().key, "b");
});

test("fromJSON tolerates missing or foreign state", () => {
  assert.equal(ScanQueue.fromJSON(undefined).size, 0);
  assert.equal(ScanQueue.fromJSON({ version: 99, items: [item("a")] }).size, 0);
  assert.equal(ScanQueue.fromJSON({ version: 1, items: "nope" }).size, 0);
});

test("msUntilReady is null for an empty queue", () => {
  assert.equal(new ScanQueue().msUntilReady(), null);
});
