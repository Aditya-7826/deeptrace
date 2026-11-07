# DeepTrace Chrome extension

The extension adds a **Scan this page** button to Chrome (or Edge). When you click it, faces in the page's images get a badge showing **Real**, **Fake**, **Uncertain** or **No face**, along with the probability that the face is fake. Click a badge to see the Grad-CAM heatmap, which shows where the model looked.

All analysis happens in **your own DeepTrace API**, which runs on your computer. The extension never scans a page on its own, and it sends nothing anywhere until you click.

> The verdicts are probabilistic estimates. They can be wrong on unseen generators, heavy compression or screenshots.

## 1. Start the API

The extension needs the API running. In the `deeptrace` folder:

PowerShell:
```powershell
uv run python -m deeptrace download-weights   # first time only
uv run python -m deeptrace serve
```

bash:
```bash
uv run python -m deeptrace download-weights   # first time only
uv run python -m deeptrace serve
```

Leave that window open. Check that it works by opening http://127.0.0.1:8000/health, which should show `{"status":"ok",...}`.

## 2. Load the extension ("Load unpacked")

1. Open Chrome and type `chrome://extensions` in the address bar, then press Enter.
   In Edge, use `edge://extensions` instead.
2. Turn on **Developer mode**. In Chrome this toggle is in the **top-right corner** of the page; in Edge it's in the **left sidebar**.
3. Click **Load unpacked**, which appears in the **top-left** once Developer mode is on.
4. In the folder dialog, select the **`extension`** folder inside `deeptrace`. That's the folder that contains `manifest.json`. Don't select `deeptrace` itself. Then click **Select Folder**.
5. A **DeepTrace** card appears. If Chrome shows an **Errors** button on it, click it and check that you selected the right folder.
6. Pin the button so it stays visible:
   1. Click the **puzzle-piece icon** (Extensions) to the right of the address bar.
   2. Click the **pin** next to DeepTrace.

   The blue magnifier icon now sits in the toolbar.

After you `git pull` a newer version, open `chrome://extensions` again and click the **circular-arrow (reload) icon** on the DeepTrace card.

## 3. Scan a page

1. Open any web page with photos of people.
2. Click the **DeepTrace icon** in the toolbar. Its tooltip reads "DeepTrace: Scan this page".
3. A badge appears in the top-left corner of each image. It shows "Scanning…" first, then the result, for example `Fake · 87% fake`:
   - The number is always the calibrated **probability that the face is fake**. That holds for "Real" badges too, so `Real · 4% fake` is a confident Real.
   - If an image has several faces, the badge shows the face most likely to be fake, plus the face count.
4. Images that load within 2 seconds of scrolling are included, so scroll down promptly if a page loads images lazily. Click the icon again later to scan any new images. Images already scanned are not sent again.
5. Click a badge to open the **heatmap** in a small window. You'll see the face crop the model saw and the whole image, with red areas marking what influenced the verdict most.
6. While a scan runs, the toolbar icon shows how many images are still waiting.

These images are skipped:
- smaller than 64×64 px;
- SVGs;
- 1×1 tracking pixels;
- duplicates of an image already scanned (same URL).

The extension cannot run on `chrome://` pages or the Chrome Web Store, because the browser blocks extensions there.

## Settings

Right-click the DeepTrace icon and choose **Options**. You can also reach Options from `chrome://extensions` → DeepTrace → **Details** → **Extension options**. The settings are:

| Setting | Default | Notes |
|---|---|---|
| API URL | `http://127.0.0.1:8000` | **Test connection** shows the model version. A non-local URL means scanned images are uploaded to that server, and a warning says so. |
| Show the probability on badges | on | Turn it off to show only Real / Fake / Uncertain. |

## Troubleshooting

| You see | What to do |
|---|---|
| "DeepTrace API not running at …: start it with `uv run python -m deeptrace serve`" | Start the API (step 1), wait until it prints `Uvicorn running`, then click the icon again. |
| Badge says **Error** | Hover over it for the reason, or click it. Common reasons: the image host refused the download, the file is over 10 MB, or the image is corrupt. |
| Badge says **No face** | The detector found no face of at least 64 px. Small thumbnails often look like this. |
| A red **!** on the icon | The page can't be scanned (a browser page or the Web Store). |
| Badges stay on "Scanning…" | The API may be busy (it handles 2 images at a time). The extension backs off and retries on "too many requests". After 3 minutes, a badge gives up with "timed out". |

## How it works

| Piece | Job |
|---|---|
| `background.js` (service worker) | Downloads the image bytes, which avoids the tainted-canvas and CORS problems a content script would hit. It then calls the API with **at most 2 requests in flight**, backs off on HTTP 429 (honouring `Retry-After`), and saves the queue in `chrome.storage.session` so a Manifest V3 worker shutdown mid-scan loses nothing. |
| `content.js` | Injected only when you click. It collects images and draws the badges in a closed shadow-DOM overlay with `position: fixed`, so it never changes the page's layout. Badges follow scrolling and resizing. `blob:` images are read in the page and handed over as `data:` URLs. |
| `lib/` | Pure logic (filtering, dedupe, queue, badge text), unit-tested with Node. |
| `heatmap.html`, `options.html` | Extension pages under a strict CSP. |

**Privacy and permissions:**
- `activeTab` and `scripting` let the extension inject the scanner when you click.
- `storage` holds the settings and the queue.
- Host permissions for `http://*/*` and `https://*/*` let the service worker download images from whatever site hosts them, and reach your API.
- Image downloads send no cookies.
- The extension contains no remote code, no `eval`, no analytics and no content scripts that run on their own.

## Tests

These unit tests use Node's built-in test runner (Node 22 or newer) and need no `npm install`:

PowerShell:
```powershell
cd extension; npm test
```

bash:
```bash
cd extension && npm test
```

There is also an end-to-end smoke test, which isn't part of CI because it needs a browser and the running API. It loads the extension into headless Microsoft Edge and checks:
- filtering and dedupe;
- `data:`, `blob:` and lazy-loaded images;
- that the page layout is unchanged;
- the heatmap popup;
- resuming after the service worker is stopped mid-scan;
- the "API not running" message.

Run it with:
```bash
node scripts/extension_smoke/smoke.mjs
```
Branded Google Chrome 137 and later ignore `--load-extension`, so on other systems set `BROWSER` to Chromium or Chrome for Testing.
