# Training DeepTrace on Kaggle (free GPU)

Kaggle gives you about **30 GPU hours per week**, with **12 hours per session**. All training
runs in `notebooks/train_kaggle.ipynb`. Your laptop only prepares data (FF++/Celeb-DF) and
receives the outputs.

> Every GPU-hour figure below is an **estimate** for planning, not a measurement. The real
> time per epoch is written to `metrics.csv` (`epoch_seconds`) in each run.

---

## 0. One-time setup

1. Create a Kaggle account and **verify your phone number**. GPUs and Internet access
   in notebooks require it.
2. *(Optional, for the CLI)* Go to **Settings → API → Create New Token** and save
   `kaggle.json` to `C:\Users\<you>\.kaggle\` (Windows) or `~/.kaggle/` (macOS/Linux).
   It is gitignored. Never commit it.
3. Build the code bundle on your laptop:

   PowerShell:
   ```powershell
   uv run python scripts/build_kaggle_bundle.py
   ```
   bash:
   ```bash
   uv run python scripts/build_kaggle_bundle.py
   ```

   This creates `artifacts/kaggle_bundle/` (wheel + configs + COMMIT).
4. Upload it as a **private** dataset named `deeptrace-bundle`:
   - **Web:** kaggle.com → **Datasets → New Dataset**. Drag in the *contents* of
     `artifacts/kaggle_bundle/` and leave visibility on **Private**. Click **Create**.
   - **CLI:** run `kaggle datasets init -p artifacts/kaggle_bundle`, edit `"id"` in the
     generated `dataset-metadata.json` to `<your-username>/deeptrace-bundle`, then run
     `kaggle datasets create -p artifacts/kaggle_bundle`. This is private by default.
     **Never** add `--public`.

   Rebuild and upload a **new version** of the bundle whenever the code changes
   (**Dataset → New Version**).

## 1. Create the notebook

1. Go to kaggle.com → **Code → New Notebook → File → Import Notebook** and upload
   `notebooks/train_kaggle.ipynb`.
2. Open the right-hand panel and set **Session options → Accelerator → GPU T4 x2**.
   If only a P100 is offered, cell 3 tells you whether the installed PyTorch still
   supports it.
3. Set **Session options → Internet → On**. Pretrained ImageNet weights download
   from the Hugging Face Hub.
4. Click **Add Input** and add `deeptrace-bundle`, plus the data listed for the run below.

## 2. Run in the background ("Save & Run All")

Interactive sessions stop when you close the tab. For real runs:

1. Edit **cell 1** (`MODE`, `CONFIG`, `CROPS`, `RESUME`).
2. Click **Save Version** (top right) → **Save & Run All (Commit)** → **Save**.
3. You can close the browser. Progress is under **View versions → the running
   version → Logs**.
4. When it finishes, outputs are on that version's **Output** tab.

## 3. Bring outputs back to your laptop

From the version's **Output** tab, download `outputs/<run_name>/` and place it at:

```
artifacts/runs/<run_name>/
    best.pt  last.pt  metrics.csv  steps.csv  curves.png
    run_info.json  config.yaml  train_summary.json  train.log
```

With the CLI (PowerShell or bash, one line):

```
kaggle kernels output <your-username>/<notebook-slug> -p artifacts/runs_download
```

Then move `artifacts/runs_download/outputs/<run_name>` to `artifacts/runs/<run_name>`.
`artifacts/` is gitignored, so weights never reach git.

## 4. Resuming after a session cut

The notebook stops cleanly before 11 h (`time_budget_hours`). It also saves `last.pt`
after every epoch, so a crash loses at most one epoch.

1. Open the notebook. Go to **Add Input → Your Work → Notebook Output** and pick
   the cut version.
2. In cell 1, set `RESUME = True` and point `PREVIOUS_OUTPUT` at that input's
   `outputs` folder.
3. **Save & Run All** again. Cell 5 copies the checkpoints back, and training
   continues exactly where it stopped. Model, optimizer, scheduler, AMP scaler and
   RNG state are all restored.

## 5. Staying inside the weekly quota

- Check the remaining hours: click your profile picture → **Settings** → the
  **Accelerator** quota bar (it also shows in the notebook's session panel).
  The quota resets weekly.
- Always use **Save & Run All**. An idle interactive session with a GPU attached
  still burns quota. Stop interactive sessions with the power button when done.
- Prove changes locally first, which costs no GPU time:
  `uv run python -m deeptrace train --config configs/debug_cpu.yaml`
- Run one job per session, in the order below. If a week runs out, carry on next week.
  `--resume` makes splitting a run safe.
- Lightning AI's free tier is the overflow option. The same commands work there:
  `pip install` the wheel and run `python -m deeptrace train ...`.

---

## The run plan (in order)

| # | When | Notebook settings (cell 1) | Inputs to attach | Est. GPU h | Bring back |
|---|------|----------------------------|------------------|-----------|------------|
| 1 | now | `MODE="prepare"` | bundle, *140k Real and Fake Faces* (xhlulu) | 0.3–0.7 | the output → **New Dataset**, **Private**, `deeptrace-crops-kaggle140k` |
| 2 | now | `MODE="train"`, `CONFIG="configs/kaggle140k_effb0.yaml"` | bundle, `deeptrace-crops-kaggle140k` | 0.3–0.8 | `artifacts/runs/effb0_kaggle140k/` |
| 3 | after FF++ + Celeb-DF access | *(laptop, no GPU)* prepare + package, see below | – | 0 | upload **private** `deeptrace-crops-ffpp` |
| 4 | after 3 | `CONFIG="configs/main_effb0.yaml"` | bundle, `deeptrace-crops-ffpp` | 0.6–1.2 | `artifacts/runs/effb0_main/` |
| 5 | after 4 | `CONFIG="configs/cmp_resnet50.yaml"` | same | 1.0–1.8 | `artifacts/runs/resnet50_cmp/` |
| 6 | after 4 | `CONFIG="configs/cmp_vit_small.yaml"` | same | 1.0–1.8 | `artifacts/runs/vits_cmp/` |
| 7 | after 4 | `CONFIG="configs/abl_noaug.yaml"` | same | 0.5–1.0 | `artifacts/runs/abl_noaug/` |
| 8 | after 4 | `CONFIG="configs/abl_frozen.yaml"` | same | 0.2–0.5 | `artifacts/runs/abl_frozen/` |
| 9 | after 3b | `CONFIG="configs/abl_fullframe.yaml"` | bundle, `deeptrace-crops-ffpp-fullframe` | 0.6–1.2 | `artifacts/runs/abl_fullframe/` |
| 10 | optional | `CONFIG="configs/rgb_fft.yaml"` | bundle, `deeptrace-crops-ffpp` | 0.7–1.4 | `artifacts/runs/rgb_fft/` |

**Total: roughly 5–10 GPU hours**, which fits within one week's quota. Set `CROPS` in cell 1
to the dataset you attached.

### Step 3: FF++ and Celeb-DF on your laptop (CPU, resumable)

FF++ and Celeb-DF are only available by access request, and must never be
redistributed. Request them now, because approval can take days:
- FaceForensics++: fill in the form linked from github.com/ondyari/FaceForensics. You
  get a download script. Download `-c c23 -t videos` for `original`, `Deepfakes`,
  `Face2Face`, `FaceSwap` and `NeuralTextures`.
- Celeb-DF v2: fill in the form linked from github.com/yuezunli/celeb-deepfakeforensics.

Unpack them to `data/raw/FaceForensics++/` and `data/raw/Celeb-DF-v2/`, then run:

PowerShell:
```powershell
uv run python -m deeptrace prepare --config configs/prepare_ffpp_celebdf.yaml
uv run python scripts/package_kaggle_dataset.py --crops-dir data/crops/ffpp_celebdf --slug deeptrace-crops-ffpp --username <your-kaggle-username>
kaggle datasets create -p artifacts/kaggle_upload/deeptrace-crops-ffpp
```
bash:
```bash
uv run python -m deeptrace prepare --config configs/prepare_ffpp_celebdf.yaml
uv run python scripts/package_kaggle_dataset.py --crops-dir data/crops/ffpp_celebdf --slug deeptrace-crops-ffpp --username <your-kaggle-username>
kaggle datasets create -p artifacts/kaggle_upload/deeptrace-crops-ffpp
```

The prepare step uses MTCNN on CPU with 2 threads. Expect it to run overnight; this is
an estimate, and the actual speed is in `prepare.log`. If it's interrupted (sleep,
crash, Ctrl+C), run the same command again, and it skips every file already done.
Keep the laptop on mains power with sleep disabled.

**3b (only for run 9):** do the same with `configs/prepare_ffpp_fullframe.yaml`
and the slug `deeptrace-crops-ffpp-fullframe`.

Before uploading, read the shortcut warnings in `prepare_report.json`. If real
and fake crops differ strongly in source resolution, JPEG quality or border padding, the
model could learn those instead of manipulation traces.
