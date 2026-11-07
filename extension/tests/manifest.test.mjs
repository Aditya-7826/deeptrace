import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(readFileSync(join(ROOT, "manifest.json"), "utf8"));

function files(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.name === "node_modules" || e.name === "tests") continue;
    const p = join(dir, e.name);
    if (e.isDirectory()) files(p, out);
    else out.push(p);
  }
  return out;
}

test("manifest v3 with only the permissions DeepTrace needs", () => {
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual([...manifest.permissions].sort(), ["activeTab", "scripting", "storage"]);
  assert.deepEqual(manifest.host_permissions, ["http://*/*", "https://*/*"]);
  assert.equal(manifest.content_scripts, undefined, "nothing runs until the user clicks");
  assert.equal(manifest.web_accessible_resources, undefined);
  assert.equal(manifest.externally_connectable, undefined);
  assert.ok(manifest.action.default_title.includes("Scan this page"));
  assert.equal(manifest.action.default_popup, undefined, "a popup would swallow the click");
});

test("strict CSP for extension pages", () => {
  const csp = manifest.content_security_policy.extension_pages;
  assert.match(csp, /script-src 'self'(;|$)/);
  assert.match(csp, /object-src 'none'/);
  assert.match(csp, /base-uri 'none'/);
  assert.doesNotMatch(csp, /unsafe-|wasm|http:\/\/|https:\/\/|\*/);
});

test("files referenced by the manifest exist", () => {
  const paths = [
    manifest.background.service_worker,
    manifest.options_ui.page,
    ...Object.values(manifest.icons),
    ...Object.values(manifest.action.default_icon),
  ];
  for (const p of paths) assert.ok(readFileSync(join(ROOT, p)).length > 0, p);
});

test("no remote code: no external scripts, eval or inline handlers", () => {
  for (const f of files(ROOT)) {
    const ext = extname(f);
    if (![".js", ".html"].includes(ext)) continue;
    const src = readFileSync(f, "utf8");
    assert.doesNotMatch(src, /<script[^>]+src=["']?(https?:)?\/\//i, f);
    assert.doesNotMatch(src, /<script(?![^>]*\ssrc=)[^>]*>/i, `${f}: inline <script>`);
    assert.doesNotMatch(src, /\son[a-z]+=["']/i, `${f}: inline event handler`);
    assert.doesNotMatch(src, /\beval\(|new Function\(|importScripts\(["']https?:/, f);
    assert.doesNotMatch(src, /\.innerHTML\s*=|insertAdjacentHTML|document\.write/, f);
  }
});
