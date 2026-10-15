# JEPA crop-classification research notebook — execution guide

`notebooks/jepa_crop_research.ipynb` is a self-contained notebook (no companion `.py` modules) that runs the complete study:

* environment audit;
* PASTIS acquisition and audit;
* parcel-level preprocessing with leakage-safe splits;
* a supervised baseline and a Random-Forest reference;
* a compact temporal JEPA with the unmodified (`random_block`) and the proposed **phenology-aware** target placement;
* linear probes and fine-tuning at nested label budgets, plus missing-observation tests;
* results and an auto-generated research summary.

| File | Purpose |
|---|---|
| `notebooks/jepa_crop_research.ipynb` | The notebook (main deliverable) |
| `notebooks/jepa_crop_research_executed_synthetic_smoke.ipynb` | The same notebook **with outputs** from my validation run (synthetic data, CPU) — for reference only |
| `docs/AUDIT_REPORT.md` | Critical audit of the handover and the frozen research design |
| `docs/VALIDATION_REPORT.md` | What was tested, what passed, what remains unverified |
| `requirements.txt`, `environment.yml` | Dependencies |

---

## 1. Setup (Windows or Linux laptop with an RTX 3050)

1. Install a recent NVIDIA driver; `nvidia-smi` should list the RTX 3050.
2. Create an environment (Python 3.10–3.12):

   ```bash
   conda env create -f environment.yml      # or: python -m venv .venv && pip install -r requirements.txt
   conda activate jepa-crop
   ```
3. **Install PyTorch with CUDA.** Use the selector at <https://pytorch.org/get-started/locally/> (choose pip, your OS, and a CUDA version supported by your driver) and run the command it gives you. Check the install with:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   It must print `True` and the RTX 3050. The notebook never installs torch itself, because the correct wheel depends on your driver.
4. Register the kernel and start Jupyter:

   ```bash
   python -m ipykernel install --user --name jepa-crop
   jupyter lab
   ```
   VS Code works too: open the notebook and select the `jepa-crop` kernel.

Start Jupyter from the folder where you want `jepa_work/` (outputs) and `data/` to live, or set the paths explicitly (§4).

## 2. Recommended order of execution

| Step | Settings (cell 0.1 or environment variables) | What it shows | Expected time |
|---|---|---|---|
| A | `PROFILE="SMOKE_TEST"`, `DATA_SOURCE="synthetic"` (the defaults) | The whole pipeline works on your machine, with no download. Section 11 also **measures the QUICK_PILOT model's speed on your GPU** and projects pilot runtime | ~3–10 min on the GPU (measured: about 2.5 min on my 1-core CPU sandbox, including the pilot-model benchmark) |
| B | Download PASTIS (§3), `DATA_SOURCE="pastis"`, still `SMOKE_TEST` | Real-data parsing, band-order check and a 60-patch end-to-end run | ~10–20 min plus download |
| C | `PROFILE="QUICK_PILOT"` | First meaningful experiment: 3 seeds × {random_block, phenology} | Estimate 2–9 h (see §6); **re-check the projection printed in cell 11.3** |
| D | `PROFILE="RESEARCH_RUN"` | 5 folds × 5 seeds × 4 policies, ablation controls | Estimate 100–200 h locally; a cloud GPU is recommended |

Always use **Run All** from the top (or run cells in order). Sections 0–11 are fast on reruns because they reuse caches and skip completed work.

Environment-variable overrides are handy for headless runs, e.g. with `papermill` or `jupyter nbconvert --execute`:

```bash
JEPA_PROFILE=QUICK_PILOT JEPA_DATA_SOURCE=pastis JEPA_PASTIS_ROOT=D:/data/PASTIS JEPA_WORK_DIR=D:/jepa_work jupyter lab
# Windows PowerShell:  $env:JEPA_PROFILE="QUICK_PILOT"; $env:JEPA_DATA_SOURCE="pastis"; jupyter lab
```

`JEPA_P_OVERRIDES='{"pretrain_epochs": 10}'` overrides individual profile entries, and `JEPA_HP_OVERRIDES` does the same for hyperparameters (e.g. the ablation `'{"token_embed": "raw_sum"}'`). Overridden runs get new config hashes, so they never mix with standard runs.

## 3. Getting PASTIS

* The official source is the PASTIS benchmark repository by V. Sainte Fare Garnot (`github.com/VSainteuf/pastis-benchmark`), which links to the Zenodo archive.
* **Before downloading, check the licence and the current archive size on the Zenodo record.** My planning figure is about 29 GB for the zip, but that is an unverified recollection.
* The notebook needs only `metadata.geojson`, `DATA_S2/`, `ANNOTATIONS/` (and optionally `INSTANCE_ANNOTATIONS/`). **Sentinel-1 / PASTIS-R is not needed.**

You have two options:

1. **Manual download (recommended).** Extract so that `PASTIS_ROOT/metadata.geojson`, `PASTIS_ROOT/DATA_S2/` and `PASTIS_ROOT/ANNOTATIONS/` exist, and set `PASTIS_ROOT`.
2. **Notebook-gated download.** Paste the official archive URL into `PASTIS_ARCHIVE_URL` and set `ALLOW_DOWNLOAD=True`.
   * The download resumes after interruption and checks free disk space.
   * If you provide `PASTIS_ARCHIVE_MD5`, the archive is verified.
   * Only the needed members are extracted.
   * The notebook contains **no hard-coded URL** by design.

Section 4 checks the file layout, array shapes and dates against what the notebook expects. Section 5.4 checks the **band order** (NIR/red must give plausible crop NDVI). On any mismatch the notebook stops with a diagnostic instead of guessing.

## 4. Configuration

All settings are in cell **0.1**: profile, data source, paths, device policy, the download gate and hyperparameters.

* **Device.** CUDA is required for `QUICK_PILOT` and `RESEARCH_RUN`. CPU is allowed only for `SMOKE_TEST` (`ALLOW_CPU`), and the notebook prints a loud warning when it uses it. It never silently falls back.
* **Precision.** bf16 autocast on Ampere GPUs such as the RTX 3050; fp16 with a GradScaler on GPUs without bf16; fp32 on CPU.
* **VRAM.** The pilot model is small (about 0.8 M encoder parameters), so 4 GB is not expected to be the bottleneck. If you hit an out-of-memory error, lower `ssl_batch_size` / `downstream_batch_size` in `PROFILES`, or lower `KEEP_DATA_ON_GPU_MAX_MB` (which moves the parcel tensor to CPU RAM). The runner stops with a diagnostic and keeps the last checkpoint.

## 5. Outputs, checkpoints and resuming

```
jepa_work/
├── cache/<source>_<n>p_<hash>/parcels.npz + manifest.json    # parcel time series (SHA-256 recorded)
├── synthetic/                                                # synthetic PASTIS-format data (only if used)
├── logs/              environment.json, package_versions.json, config snapshots, benchmark JSON
├── runs/<source>__<PROFILE>/
│   ├── split_fc<k>.json, norm_stats_fc<k>.json               # split membership + training-only statistics
│   ├── _sanity/                                              # artifacts of Section 10 tests (recreated each time)
│   └── <run_id>__<config_hash>/
│       ├── config.json  status.json  log.jsonl  metrics.json
│       ├── ckpt_last.pt  (SSL; removed for completed downstream runs)
│       ├── ckpt_best.pt  (downstream: best validation macro-F1)
│       ├── encoder_final.pt  (SSL: EMA target + context encoders)
│       └── test_predictions*.npz
├── results/<namespace>/   sanity_tests.json, timings.json, RESEARCH_SUMMARY.md
├── tables/<namespace>/    CSV tables (main results, paired differences, robustness, per-class F1, runtime)
└── figures/<namespace>/   PNG figures
```

* **Resume.** Re-run the interrupted cell (after re-running Sections 0–11 if the kernel restarted).
  * Completed runs are skipped.
  * Interrupted runs continue from `ckpt_last.pt`, which holds the model, optimiser, scaler, step and all RNG states. On CPU the resumed run was verified to be bit-identical to an uninterrupted one.
  * `Ctrl-C` / "Interrupt kernel" saves a checkpoint first.
  * Periodic checkpoints are written every 10 minutes and at every epoch or evaluation.
* **Changing a setting** creates new run directories (new config hash). Old results are kept, never overwritten or mixed.
* **Nothing valuable is deleted automatically.**
  * Corrupt or incompatible checkpoints raise an error; you must move them yourself.
  * The only automatic deletions are regenerable intermediates: preprocessing shards after a verified merge, the `_sanity` test folder, and the redundant `ckpt_last.pt` of a *completed* downstream run (its best checkpoint is kept).

## 6. Runtime and storage (planning estimates — the notebook measures the real numbers)

Assumptions behind these estimates:

* about 70 k training parcels (unverified until your cache is built);
* up to 61 dates × 10 bands;
* batch 256 and bf16;
* encoder d=128 with 4 layers, predictor 64-wide with 3 layers;
* an assumed 60–120 ms per SSL step on an RTX 3050 Laptop GPU.

| Stage | RTX 3050 (local), low / typical / high |
|---|---|
| Environment setup | 0.3 / 0.5 / 1 h |
| PASTIS download | 0.5 / 1.5 / 4 h (network-bound) |
| Parcel extraction (one pass over all patches) | 10 / 20 / 40 min |
| Sanity tests + benchmark | 5 / 8 / 15 min |
| Supervised (9 runs) + RF (9 runs) | 0.4 / 0.8 / 1.5 h |
| JEPA pretraining (6 runs × 40 epochs) | 1 / 2 / 3 h |
| Linear probes (9 runs) + fine-tuning (18 runs) | 0.4 / 0.8 / 1.5 h |
| **QUICK_PILOT total (excl. download)** | **≈ 2 / 4 / 9 h** |
| **RESEARCH_RUN total** | **≈ 100 / 150 / 200+ h** |

Cloud option: Colab or Kaggle GPUs vary per session, so the notebook detects and logs the actual GPU. For a model this small, a T4 may be only about 1–1.5× faster than the laptop; an L4 or A100 is roughly 3–6× faster. The cloud is mainly useful for `RESEARCH_RUN`, because it can run unattended.

Cell **11.3** replaces these planning numbers with projections from throughput **measured on your machine**. Each training epoch prints its measured time and an estimate of the remaining time.

| Storage item | Size |
|---|---|
| PASTIS archive (zip) | ~29 GB (unverified; check Zenodo) |
| Extracted needed members | 25–45 GB |
| Parcel cache | 0.1–0.5 GB (shards are deleted after merge) |
| Checkpoints | QUICK_PILOT ≤ 0.5 GB; RESEARCH_RUN ≈ 2–3 GB |
| Logs, predictions, figures | ≤ 0.5 GB |
| **Recommended free space** | **≥ 80 GB if you keep the zip; ≥ 50 GB if you delete it after extraction** |

## 7. Common errors

| Message | Fix |
|---|---|
| `CUDA is not available and ALLOW_CPU=False` | A CPU-only torch is installed or the driver is missing. Reinstall torch with CUDA (§1.3) and check `nvidia-smi` |
| `PASTIS not found: missing …metadata.geojson` | Set `JEPA_PASTIS_ROOT` to the folder that directly contains `metadata.geojson` (the error lists candidate folders) |
| `expected S2 array (T, 10, H, W)` / `Band-order check FAILED` | Your copy differs from the assumed layout. Check the dataset documentation and adjust `BAND_NAMES`, `RED_IDX` and `NIR_IDX` |
| `Existing cache … built with a different configuration` | Move the old cache folder aside or use a new `WORK_DIR` |
| `CorruptCheckpointError` | Usually a crash mid-write on an old file. Move or rename the named file and re-run; the run restarts from scratch |
| Gate G3 FAIL with "shortcut" in the summary | Target features mostly encode acquisition date. Keep `token_embed="ln_content_learned_date"`, lower `ssl_lr`, and inspect `target_date_var_ratio` in the SSL logs |
| `DECISION GATE 1 FAILED` | A sanity test failed; the failing test is printed with details. Do not run experiments until it passes |
| `CUDA out of memory` | Lower the batch sizes in `PROFILES` or `KEEP_DATA_ON_GPU_MAX_MB` |
| `Non-finite SSL loss` | Lower `ssl_lr`; check the input audit for NaN/inf; try `AMP_MODE="off"` to rule out precision issues |
| Windows: Random Forest is slow or hangs | Set `n_jobs` lower in `run_rf` (joblib process pool) or set `run_rf=False` in the profile |
