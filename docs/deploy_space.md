# Deploying the Hugging Face Space

The demo runs on a **free "CPU basic" Space** (2 vCPU, 16 GB RAM). The Space holds only two small files from this repo:

- `space/README.md`: Space settings in the front matter, and the page text.
- `space/Dockerfile`: downloads the tagged DeepTrace release, installs it with `uv sync --frozen`, and runs `space/app.py`.

The app code, the example images and the weights list all come from the **GitHub release tag**. You redeploy by changing the tag; there's no code to copy.

<details>
<summary>Why a Docker Space and not the plain Gradio SDK?</summary>

A Gradio-SDK Space installs packages with pip. pip cannot install DeepTrace, because `facenet-pytorch` 2.6.0 declares `torch<2.3`, `numpy<2` and `pillow<10.3`, while DeepTrace needs torch ≥ 2.4. We relax those pins with a `uv` override, which only `uv` understands. The Docker Space installs with `uv sync --frozen`, so it runs exactly the versions in `uv.lock`, the ones the tests passed on. The app is still a Gradio app.
</details>

## Before you start

1. **The release must exist.** Phase 8 creates GitHub Release **v1.0.0** with the model files and a `weights.json` listing their SHA-256 sums.
   - Check that it exists: https://github.com/Aditya-7826/deeptrace/releases/tag/v1.0.0.
   - Without it, the Space builds but shows "Model weights could not be downloaded".
2. **The GitHub repo must be public,** so the Space can download the source and the weights without a token.
3. **You need a Hugging Face account** (free) to create the Space.
4. **For option B only:** a Hugging Face access token with **write** permission. Create one at https://huggingface.co/settings/tokens: **Create new token** → **Write** → copy it.

## 1. Create the Space

1. Go to https://huggingface.co/new-space.
2. Fill in the form:

   | Field | Value |
   |---|---|
   | Owner | your username |
   | Space name | `deeptrace` |
   | License | `mit` |
   | Select the Space SDK | **Docker**, then template **Blank** |
   | Space hardware | **CPU basic · 2 vCPU · 16 GB · FREE** |
   | Visibility | **Public** |

3. Click **Create Space**. You land on an empty Space page with instructions.

## 2. Add the two files

Pick **one** of these options.

### Option A: in the browser (no git needed)

1. On the Space page, open the **Files** tab.
2. Click **+ Contributions** (top right) → **Upload files**.
3. Drag in `space/README.md` and `space/Dockerfile` from your local `deeptrace` folder. Uploading `README.md` replaces the placeholder README; that's expected.
4. Write a commit message such as `Deploy DeepTrace v1.0.0`, then click **Commit changes to main**.

### Option B: with git

Replace `<user>` with your Hugging Face username. When git asks for a password, paste your **write token**.

PowerShell:
```powershell
cd $HOME
git clone https://huggingface.co/spaces/<user>/deeptrace deeptrace-space
Copy-Item deeptrace\space\README.md, deeptrace\space\Dockerfile deeptrace-space\ -Force
cd deeptrace-space
git add README.md Dockerfile
git commit -m "Deploy DeepTrace v1.0.0"
git push
```

bash:
```bash
cd ~
git clone https://huggingface.co/spaces/<user>/deeptrace deeptrace-space
cp deeptrace/space/README.md deeptrace/space/Dockerfile deeptrace-space/
cd deeptrace-space
git add README.md Dockerfile
git commit -m "Deploy DeepTrace v1.0.0"
git push
```

These commands assume the `deeptrace` repo is in your home folder; adjust the path if it isn't.

## 3. Watch it build

1. The Space page shows **Building**. Click **Logs** (next to the status badge) → **Build** to follow it. The first build takes about 5–10 minutes, mostly downloading CPU PyTorch.
2. When the status turns **Running**, open **Logs** → **Container**. You should see lines like:
   ```text
   model_fp32.onnx: downloaded, checksum ok
   * Running on local URL:  http://0.0.0.0:7860
   ```
3. Open the **App** tab, click an example such as `aldrin.jpg`, and check that you get a verdict and two heatmaps.

## 4. Link it from the GitHub README

The README's "Open in Spaces" badge points at `https://huggingface.co/spaces/Aditya-7826/deeptrace`. If your Hugging Face username differs from `Aditya-7826`, change that URL in `README.md`.

## Updating to a new release

1. Publish the new GitHub release (for example `v1.1.0`) with its weights.
2. On the Space, go to **Settings** → **Variables and secrets** → **New variable**. Set name `DEEPTRACE_REF` and value `v1.1.0`, then save.
3. Go to **Settings** → **Factory rebuild**. The Space passes the variable to the Docker build, so no files change.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Build log shows `404` on `archive/refs/tags/v1.0.0.tar.gz` | The tag doesn't exist yet, or the repo is private. Create the release and make the repo public. |
| The app says "Model weights could not be downloaded: this code version has no released weights yet" | `weights.json` in that tag has no assets. Run `scripts/release_weights.py` before tagging (Phase 8 does this). |
| "checksum mismatch" | A release asset was replaced after tagging. Re-upload the original files or cut a new tag. |
| Space shows **Runtime error** with `Killed` | Out of memory (unlikely: the app peaks at about 0.9 GB locally). Check the Container log. |
| Space sleeps after 48 h without visitors | Normal on the free tier. The next visitor wakes it, which takes about a minute. |

## Running the same app locally

You don't need Docker for this:

PowerShell:
```powershell
uv sync --extra space
uv run python -m deeptrace download-weights
uv run python space/app.py
```

bash:
```bash
uv sync --extra space
uv run python -m deeptrace download-weights
uv run python space/app.py
```

Then open http://127.0.0.1:7860. Run these commands from the `deeptrace` folder: the app looks for `weights/` there, or in `DEEPTRACE_WEIGHTS_DIR` if that's set.
