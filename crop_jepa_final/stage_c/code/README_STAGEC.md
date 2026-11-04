# Crop-JEPA Stage C on the SPIT GPU cluster

Stage C lives in its own folder, `~/crop_jepa_stageC`. It **reads** three Stage B files and never writes into
`~/crop_jepa_cluster`:
`outputs_stageB/cache/images_192.{npy,csv}`, `outputs_stageB/splits.csv` (Split A), `outputs_stageB/stats/seed_level_primary.csv`.

## 0. Upload (Windows PowerShell, on the laptop)
```powershell
scp -r "C:\Users\kalhan\Desktop\crop_jepa_cluster\crop_jepa_stageC" <username>@cluster.gpu.spit.ac.in:~/
```

## 1. First run (cluster shell)
```bash
cd ~/crop_jepa_stageC
sed -i 's/\r$//' *.sh *.sbatch *.env *.py     # safety: strip Windows line endings if any slipped in
ls ~/crop_jepa_cluster/outputs_stageB/cache/images_192.npy ~/crop_jepa_cluster/outputs_stageB/splits.csv   # must exist
bash run_stageC.sh
```
This submits a CPU prep job (Python env in `./venv`, read-only checks of the Stage B inputs, optional DINOv2 download),
then one GPU job (1 GPU, 3 worker processes sharing it; 32 CPUs, 64 GB, 72 h). The GPU job runs:

1. **smoke test** (generated data, a few minutes),
2. **Phase 0** (≈ 30–40 min, validation data only): context-token audit → candidate budget K; two ijepa runs (seed 900,
   9,030 steps) uncapped vs capped at K → protocol decision; ExG-dispersion tertile cut points; power simulation; then it
   writes the **frozen pre-registration** `outputs_stageC/phase0/stats/preregistration_stageC.json` + `.sha256`.
3. **Phase-0 sanity gate** (automatic): both Phase-0 runs have 0 non-finite steps, uncapped k-NN val F1 ≥ 0.30
   (Stage B: 0.46–0.47) and RankMe ≥ 20. If the gate passes, the job writes `outputs_stageC/PREREG_CONFIRMED`
   (an internal freeze record: the hash and the release time; there is no external registration) and continues
   straight into the main study. If it fails, the job stops before any main run.

## 2. Main study (same job, no action needed)
The job then runs **core → splitB → vits → bridge**, each as: a prepare pass, 3 workers sharing the GPU (`NUM_GPUS` x `WORKERS_PER_GPU`), and a report pass
(statistics, figures, LaTeX tables). It resubmits itself until everything is finished, then writes
`outputs_stageC/FINAL_REPORT.md`. Estimated time: ≈ 45–52 h on 1 GPU (≈ 24–29 h with `NUM_GPUS=2`); an estimate, not a measurement. If the wall time runs out, every run stops at an epoch boundary with a checkpoint and the job re-submits itself (duplicate-job guards prevent two Stage C jobs running at once).

Every phase refuses to start if the pre-registration file changed, if it was not released by the gate, or if the code differs from
the registered code hash (`CJ_ALLOW_CODE_CHANGE=1` in `config_stageC.env` only for a deliberate, documented bug fix; it
is logged in `outputs_stageC/DEVIATIONS.md`).

## 3. Watch
```bash
bash monitor_stageC.sh          # queue, state markers, finished runs per phase, latest training lines
tail -f outputs_stageC/logs/pipeline.log
```
If something stops, `outputs_stageC/state/STOP` says why. Fix the cause, then `bash run_stageC.sh` (finished runs are
skipped; unfinished runs resume from their last checkpoint).

## 4. Collect
```bash
bash package_results.sh          # stageC_results.zip + .sha256 (add 'features' to include float16 features)
```
```powershell
scp <username>@cluster.gpu.spit.ac.in:~/crop_jepa_stageC/stageC_results.zip "C:\Users\kalhan\Desktop\"
Get-FileHash "C:\Users\kalhan\Desktop\stageC_results.zip" -Algorithm SHA256    # must equal the .sha256 on the cluster
```

## Where things are
| path | content |
|---|---|
| `outputs_stageC/phase0/stats/` | `stageC_locked.json` (Phase-0 decisions), `preregistration_stageC.json` (+ `.sha256`), `power_analysis.json` |
| `outputs_stageC/<phase>/runs/` | one folder per run: config, train log, `eval.json`, `preds.npz`, `features_*.npy`, encoder |
| `outputs_stageC/<phase>/stats/` | `gate_G1_stageC_<phase>.json` (decision), `paired_comparisons.csv`, `moderation.csv`, ... |
| `outputs_stageC/<phase>/figures`, `paper_tables` | PNG figures, CSV + LaTeX tables |
| `outputs_stageC/FINAL_REPORT.md` | cross-phase report: decision, all pre-registered tests, meta-analysis, bridge, allowed claims |
| `outputs_stageC/logs/` | `pipeline.log` (the readable story) and one log per step / worker |

## Files
`crop_jepa_stageC.py` (all science; `Crop_JEPA_StageC.ipynb` is generated from it by `make_notebook.py`),
`config_stageC.env` (cluster settings only), `stageC_common.sh` (per-phase settings = `PHASE_PLAN` in the script),
`stageC_prep.sbatch`, `stageC_job.sbatch`, `submit_gpu.sh`, `run_stageC.sh`, `monitor_stageC.sh`,
`check_complete_stageC.py`, `stageC_final_report.py`, `package_results.sh`, `setup_env.sh`.

## Settings you may change (in `config_stageC.env`)
`NUM_GPUS` (1; 2 is faster but needs both GPUs free at once), `WORKERS_PER_GPU`, `CPUS_PER_TASK`, `MEM_GB`, `WALL_TIME` (must not exceed the partition limit: `sinfo -p general -h -o "%l"`), `ENABLE_DINOV2`. The experiment itself
(arms, seeds, steps, splits) is fixed by the pre-registration and cannot be changed from the config.
