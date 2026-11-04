# Crop-JEPA Stage C: run guide, start to finish

Every command you need, in order. **Laptop** commands are Windows PowerShell. **Cluster** commands are bash after `ssh`.

There is **no OSF step**. Phase 0 still freezes the analysis plan with a SHA-256 hash before any main run (an *internal freeze*). An automatic sanity gate then releases the main study in the same job, so you don't have to do anything between Phase 0 and the main study.

Overview:

| step | where | what | time |
|---|---|---|---|
| 0 | laptop + cluster | **you are here**: replace pending 2-GPU job 2145 with the 1-GPU job (section 0) | 5 min |
| 1 | laptop | upload the Stage C folder | 1 min |
| 2 | cluster | pre-flight checks (read only) | 2 min |
| 3 | cluster | `bash run_stageC.sh`: CPU prep job → one GPU job that runs the smoke test, Phase 0, the sanity gate, then core → splitB → vits → bridge → final report | ≈ 45–52 h on 1 GPU (estimate) + queue wait |
| 4 | cluster | monitor (optional, as often as you like) | – |
| 5 | cluster | if something stops: read `state/STOP`, fix, `bash run_stageC.sh` | – |
| 6 | cluster + laptop | package and download the results | 5 min |

Ground rules:
- Never edit `crop_jepa_stageC.py` after Phase 0. The frozen pre-registration stores its SHA-256, and the main phases refuse to start if the code changed.
- In the paper, say *"the analysis plan was frozen (SHA-256, timestamped) in Phase 0, before any main-study run; it was not externally registered"*. **Don't** call it OSF-preregistered.
- Never touch `~/crop_jepa_cluster` (Stage B). Stage C only reads three files from it.
- Don't run training on the login node. Everything below goes through `sbatch`.
- Cancel only your own job by its ID (`scancel <JOBID>`). Never use `scancel -u`.

---

## 0. Replace pending job 2145 with the 1-GPU job (do this once)

Your situation:
```
2145   cjC   PENDING   (Resources)       state: dinov2_repo prep.ok
```
- **`prep.ok` and `dinov2_repo`**: the CPU prep job **succeeded**. The environment is built, the Stage B inputs checked out, and DINOv2 is downloaded. All of it is reused; it won't run again.
- **Job 2145** asks for **2 GPUs, 64 CPUs, 128 GB, 36 h**. Both GPUs are held by other users' 1-GPU jobs, so it can only start once both are free at the same time.
- **The new configuration** asks for **1 GPU, 32 CPUs, 64 GB, 72 h**. This is the same per-GPU setup as Stage B's job 2070, which ran on this node and finished cleanly. Slurm keeps a copy of a job's script and resources **from submission**, so 2145 must be cancelled and resubmitted to pick up the change. Nothing is lost: it hasn't started, and `outputs_stageC/` (state markers, logs, venv, DINOv2) is kept.

**0a. Laptop (PowerShell): upload the changed files.** `crop_jepa_stageC.py` is unchanged; don't upload it again.
```powershell
cd "C:\Users\kalhan\Desktop\crop_jepa_cluster\crop_jepa_stageC"
scp config_stageC.env stageC_job.sbatch README_STAGEC.md STAGE_C_RUN_GUIDE.md Crop_JEPA_StageC.ipynb make_notebook.py <username>@cluster.gpu.spit.ac.in:~/crop_jepa_stageC/
```

**0b. Cluster: check, then replace.**
```bash
cd ~/crop_jepa_stageC
sed -i 's/\r$//' *.sh *.sbatch *.env

# --- check that the new files are in place ---
grep -E "^export (NUM_GPUS|WORKERS_PER_GPU|CPUS_PER_TASK|MEM_GB|WALL_TIME)=" config_stageC.env
#   expect NUM_GPUS=1, WORKERS_PER_GPU=3, CPUS_PER_TASK=32, MEM_GB=64, WALL_TIME="3-00:00:00"
grep -E "^#SBATCH --(gres|mem|cpus|time)" stageC_job.sbatch   # expect gpu:1, 64G, 32, 3-00:00:00
grep -c "Phase-0 sanity gate" stageC_job.sbatch               # >= 1
grep -c "other_cjC_jobs" stageC_job.sbatch                    # >= 3 (duplicate-job guards)
sha256sum crop_jepa_stageC.py      # 859588be2b085008602897f4a5d10210ce25cbaeb850846c63866fbf7870324e
ls outputs_stageC/state/           # dinov2_repo prep.ok   (reused)
ls venv/.ready                     # exists                (reused)

# --- partition limit: the 72 h request must fit ---
sinfo -p general -h -o "%l"        # e.g. 7-00:00:00 -> fine. If it prints LESS than 3-00:00:00, first edit WALL_TIME
                                   # in config_stageC.env to that limit (or "1-12:00:00"); the job then chains itself.
scontrol show partition general | grep -oE "MaxTime=[^ ]+|MaxMemPerNode=[^ ]+|MaxCPUsPerNode=[^ ]+"

# --- replace the job ---
squeue -u $USER                    # cjC 2145 PENDING
scancel 2145                       # cancels ONLY job 2145
squeue -u $USER                    # cjC is gone (run_stageC.sh refuses while a cjC job is still listed)
bash run_stageC.sh                 # -> "Submitted Stage C GPU job NNNN (prep already done)."
squeue -u $USER -o "%.8i %.6j %.9T %.11l %.6C %.8m %.15b %R"   # TIME_LIMIT 3-00:00:00, CPUS 32, MIN_MEMORY 64G, gres/gpu:1
```
If `run_stageC.sh` prints `Submitted CPU prep job ...` instead of `(prep already done)`, then `prep.ok` or `venv/.ready` is missing. Don't worry: the prep job is idempotent and reuses the existing venv.

If sbatch rejects the time limit (`Requested time limit is invalid`), nothing was submitted. Set `WALL_TIME` to the partition limit and run `bash run_stageC.sh` again.

Then go to **section 4 (Monitoring)**. Sections 1–3 are for a fresh start.

**While it waits for a GPU:**
```bash
squeue                              # everyone's jobs: who holds the GPUs and until when (TIME / TIME_LIMIT)
squeue --start -u $USER             # Slurm's estimated start time (may say N/A)
```

---

## 1. Upload (laptop, PowerShell)

First, check whether a Stage C folder already exists on the cluster:
```powershell
ssh <username>@cluster.gpu.spit.ac.in "ls -la ~/crop_jepa_stageC 2>/dev/null || echo 'no stageC folder yet - OK'"
```
- If it prints `no stageC folder yet - OK`, upload:
  ```powershell
  scp -r "C:\Users\kalhan\Desktop\crop_jepa_cluster\crop_jepa_stageC" <username>@cluster.gpu.spit.ac.in:~/
  ```
- If the folder already exists and contains `outputs_stageC`, **don't upload over it**. Ask first.

---

## 2. Pre-flight checks (cluster)

```bash
ssh <username>@cluster.gpu.spit.ac.in
cd ~/crop_jepa_stageC

# 2a. strip Windows line endings (the files are already LF; this is only a safety net)
sed -i 's/\r$//' *.sh *.sbatch *.env *.py

# 2b. all files present?
ls
# expect: Crop_JEPA_StageC.ipynb README_STAGEC.md STAGE_C_RUN_GUIDE.md check_complete_stageC.py config_stageC.env
#         crop_jepa_stageC.py make_notebook.py monitor_stageC.sh package_results.sh run_stageC.sh setup_env.sh
#         stageC_common.sh stageC_final_report.py stageC_job.sbatch stageC_prep.sbatch submit_gpu.sh

# 2c. code hash: write this down; it must match what goes into the pre-registration later
sha256sum crop_jepa_stageC.py

# 2d. the Stage B inputs Stage C reads (must all exist)
ls -l ~/crop_jepa_cluster/outputs_stageB/cache/images_192.npy \
      ~/crop_jepa_cluster/outputs_stageB/cache/images_192.csv \
      ~/crop_jepa_cluster/outputs_stageB/splits.csv \
      ~/crop_jepa_cluster/outputs_stageB/stats/seed_level_primary.csv

# 2e. disk space: Stage C needs a few GB (encoders, float16 features, logs); keep >= 20 GB free to be safe
df -h ~

# 2f. cluster resources: partition 'general' should be up with GPUs listed
sinfo -o "%P %a %l %D %c %m %G"

# 2g. nothing of yours is already queued
squeue -u $USER
```
If any file in 2d is missing, **stop**: the prep job would fail on it anyway.

Settings live in `config_stageC.env`. The defaults are 1 GPU, 3 workers on it, 32 CPUs, 64 GB RAM and a 72 h wall time, the same per-GPU setup as Stage B's successful job 2070. `NUM_GPUS=2` also works and roughly halves the time, but it needs both GPUs free at once.
```bash
nano config_stageC.env      # only if you need to change something; Ctrl+O, Enter, Ctrl+X
```

---

## 3. Submit: one command runs everything (cluster)

```bash
cd ~/crop_jepa_stageC
bash run_stageC.sh
```
Expected output on a fresh start:
```
Submitted CPU prep job 12345.
Submitted Stage C GPU job 12346 (starts after the prep job succeeds).
```
If prep already succeeded, as in your case, you get `Submitted Stage C GPU job NNNN (prep already done).`

What happens:
1. **Prep job `cjC_prep`** (CPU, ≤ 2 h, usually 10–20 min):
   - creates `./venv` with torch 2.8.0 and torchvision 0.23.0;
   - runs read-only checks on the Stage B cache and Split A (11115 / 2380 / 2381);
   - downloads DINOv2 (optional);
   - writes `state/prep.ok`.
2. **GPU job `cjC`** (1 GPU, 3 worker processes sharing it). It runs these steps in order, with no action from you:
   1. **Smoke test** on generated data, a few minutes. It writes `state/smoke.ok`.
   2. **Phase 0** on validation data only, about 30–40 min. It runs the context-token audit (which sets the cap K) and two ijepa runs at seed 900, uncapped vs capped, which decide the protocol. It then computes the ExG-dispersion cut points, runs the power simulation, and writes the **frozen pre-registration** with its SHA-256. It writes `state/phase0.ok`.
   3. **Phase-0 sanity gate** (automatic, instead of a manual review). Both Phase-0 runs must have 0 non-finite steps, uncapped k-NN val F1 ≥ 0.30 (Stage B ijepa at the same 9,030 steps: 0.463–0.468), and uncapped RankMe ≥ 20 (Stage B: 47–51).
      - If it passes, the job writes `outputs_stageC/PREREG_CONFIRMED`, the internal freeze record: hash, freeze time, release time.
      - If it fails, the job stops **before any main run** (section 5).
      - The masking checks were already hard gates inside Phase 0: any `[FAIL]` stops Phase 0 itself.
   4. **Main study**, four phases. Each phase is a prepare pass, then 3 workers sharing the GPU, then a report pass:

      | phase | runs | what it tests |
      |---|---|---|
      | core | 6 arms × 20 seeds (100–119) + random_init + supervised (10 seeds) + DINOv2 | H1–H4 (Holm), E1–E2 equivalence, H5 moderation |
      | splitB | 5 arms × 10 seeds on a new group split | replication on a different split |
      | vits | ijepa, veg_pos × 10 seeds, ViT-S/16 | does it hold at larger capacity |
      | bridge | ijepa, veg_pos × 10 seeds, Stage B protocol | link to Stage B (pooled with Stage B seeds 0–5) |

      After each phase it writes `state/<phase>.done`.
   5. **Final report**: `outputs_stageC/FINAL_REPORT.md` and `state/DONE`.

If the 72 h wall time runs out, the job **re-submits itself**. Finished runs are skipped, and unfinished runs resume from their checkpoint. It gives up after 6 attempts, or if an attempt made no progress.

Before training, every phase re-checks four things and refuses to run if any fails: the pre-registration hash, the presence of `PREREG_CONFIRMED`, the phase plan, and the code hash.

Logs: prep goes to `outputs_stageC/logs/cjC_prep_<JOBID>.out` / `.err`, the GPU job to `outputs_stageC/logs/cjC_<JOBID>.out` / `.err`, and the readable story to `outputs_stageC/logs/pipeline.log`.

If the GPU job shows `(DependencyNeverSatisfied)` or disappears right after the prep job, the prep job failed. Look at `tail -n 50 outputs_stageC/logs/cjC_prep_*.out outputs_stageC/logs/setup_env.log`, fix the cause, then run `bash run_stageC.sh` again.

### 3b. Look at Phase 0 once it's done (optional, about 40 min after the GPU job starts)
The job doesn't wait for you, but it's worth a look:
```bash
cd ~/crop_jepa_stageC
grep -A3 "Phase-0 sanity gate" outputs_stageC/logs/pipeline.log
cat outputs_stageC/PREREG_CONFIRMED
cat outputs_stageC/phase0/stats/stageC_locked.json
grep -E "\[(PASS|FAIL)\]|KL\(|Mean z-scored|K = floor|Phase-0 decision" outputs_stageC/logs/step2_phase0.log
python3 -c "import json;print(json.load(open('outputs_stageC/phase0/stats/preregistration_stageC.json'))['code_sha256'])"
sha256sum crop_jepa_stageC.py            # the two hashes must be identical
```
What to expect:
- every check is `[PASS]`;
- KL(veg_pos‖pos_matched) and KL(veg_neg‖neg_matched) are below 0.005 (≈ 0.0006 on the laptop test);
- the ExG z order is veg_pos (≈ +0.43) > ijepa (≈ 0) > veg_neg (≈ −0.38);
- the cap is 12;
- the protocol is `C (fixed K)` or `B (...)`; both are valid outcomes of the fixed rule;
- uncapped k-NN is ≈ 0.46–0.47.

Keep a copy of the frozen plan on your laptop (PowerShell). It's your timestamped evidence that the plan was fixed before the main runs:
```powershell
mkdir "C:\Users\kalhan\Desktop\stageC_prereg" -Force
scp "<username>@cluster.gpu.spit.ac.in:~/crop_jepa_stageC/outputs_stageC/phase0/stats/*.json*" "C:\Users\kalhan\Desktop\stageC_prereg\"
scp <username>@cluster.gpu.spit.ac.in:~/crop_jepa_stageC/outputs_stageC/PREREG_CONFIRMED "C:\Users\kalhan\Desktop\stageC_prereg\"
```

---

## 4. Monitoring (cluster, any time; read only)

```bash
cd ~/crop_jepa_stageC
bash monitor_stageC.sh
```
It shows:
- the queue;
- the state markers;
- the Phase-0 decision;
- the last 12 pipeline-log lines;
- finished runs per phase;
- the latest training line of every worker;
- the GPUs, when run inside the job.

More detail:
```bash
tail -f outputs_stageC/logs/pipeline.log                    # the readable story
tail -n 5 outputs_stageC/logs/phase_core_worker*.log        # per worker (replace core with the current phase)
venv/bin/python check_complete_stageC.py all                # runs finished / expected per phase
ls outputs_stageC/core/runs/ | head                         # one folder per run
cat outputs_stageC/core/runs/*/error.txt 2>/dev/null        # errors of failed runs, if any
srun --jobid=<JOBID> --overlap --pty nvidia-smi             # GPU use inside the running job
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS   # after a job ended
```

State markers in `outputs_stageC/state/`:

| file | meaning |
|---|---|
| `prep.ok` | environment and Stage B inputs OK |
| `smoke.ok` | smoke test passed |
| `phase0.ok` | Phase 0 done, pre-registration frozen |
| `outputs_stageC/PREREG_CONFIRMED` (not in `state/`) | sanity gate passed, main phases released (internal freeze record) |
| `core.done`, `splitB.done`, `vits.done`, `bridge.done` | phase finished, statistics written |
| `DONE` | everything finished, `FINAL_REPORT.md` written |
| `STOP` | the pipeline stopped on purpose; `cat` it to see why |
| `attempts` | number of automatic re-submissions |

---

## 5. When something stops

```bash
cat outputs_stageC/state/STOP
```

| STOP message | do this |
|---|---|
| `Phase 0 is complete and the pre-registration is FROZEN ... Register it on OSF` | you're running the **old** job script; do section 0 (upload the new files, then `bash run_stageC.sh`) |
| `Phase-0 sanity gate FAILED ...` | no main run has started. Send me `grep -A3 "sanity gate" outputs_stageC/logs/pipeline.log` and `cat outputs_stageC/phase0/stats/stageC_locked.json`. **Don't** just rerun it: the frozen plan would be the same |
| `prep job has not completed` | `bash run_stageC.sh` (it resubmits the prep job) |
| `PyTorch cannot use the GPU(s)` | `cat outputs_stageC/logs/setup_env.log`; ask the cluster admins about CUDA. Don't install other torch versions yourself |
| `smoke test FAILED` | `tail -n 60 outputs_stageC/logs/step1_smoke.log`, then ask (nothing expensive ran) |
| `Phase 0 FAILED` | `tail -n 60 outputs_stageC/logs/step2_phase0.log`, then ask |
| `phase X prepare pass FAILED` | `tail -n 60 outputs_stageC/logs/phase_X_prepare.log`. If it says *code hash differs* or *does not match its frozen hash*, a file was changed after Phase 0: restore it, don't bypass |
| `no progress since the previous attempt` | `tail -n 60 outputs_stageC/logs/phase_*_worker*.log`, then ask |
| `gave up after 6 attempts` | same as above |
| `final report FAILED` | `tail -n 60 outputs_stageC/logs/step4_final.log`, then ask (all runs are kept) |

After fixing the cause, run:
```bash
bash run_stageC.sh       # clears STOP and resubmits; finished work is never repeated
```
To cancel a job you started (by ID only):
```bash
squeue -u $USER
scancel <JOBID>
```
Cancelling is safe. A later `bash run_stageC.sh` continues from the checkpoints.

**Only for a real bug found after Phase 0:**
1. Set `CJ_ALLOW_CODE_CHANGE=1` in `config_stageC.env`.
2. Make the fix and run `bash run_stageC.sh`.
3. The change is logged in `outputs_stageC/DEVIATIONS.md` and must be reported in the paper.

---

## 6. Collect the results

When `bash monitor_stageC.sh` shows `DONE`:
```bash
cd ~/crop_jepa_stageC
cat outputs_stageC/FINAL_REPORT.md | head -80
bash package_results.sh               # -> stageC_results.zip + stageC_results.zip.sha256 (no checkpoints, no features)
# bash package_results.sh features    # optional: also the float16 features (larger; lets you recompute everything offline)
cat stageC_results.zip.sha256
```
On the laptop (PowerShell):
```powershell
scp <username>@cluster.gpu.spit.ac.in:~/crop_jepa_stageC/stageC_results.zip "C:\Users\kalhan\Desktop\"
Get-FileHash "C:\Users\kalhan\Desktop\stageC_results.zip" -Algorithm SHA256     # must equal the .sha256 on the cluster
```

What's inside:

| path | content |
|---|---|
| `outputs_stageC/FINAL_REPORT.md` | decision (P-A … P-D), all pre-registered tests, meta-analysis, bridge, allowed and forbidden claims |
| `outputs_stageC/final_summary.json`, `consistency_table.csv` | machine-readable summary |
| `outputs_stageC/<phase>/stats/gate_G1_stageC_<phase>.json` | per-phase decision |
| `outputs_stageC/<phase>/stats/paired_comparisons.csv`, `moderation.csv` | test statistics |
| `outputs_stageC/<phase>/figures/`, `paper_tables/` | PNG figures, CSV and LaTeX tables |
| `outputs_stageC/phase0/stats/` | locked decisions, pre-registration, power analysis |
| `outputs_stageC/logs/` | every log |

---

## Quick reference

```bash
ssh <username>@cluster.gpu.spit.ac.in
cd ~/crop_jepa_stageC
bash run_stageC.sh            # start / continue (safe to repeat)
bash monitor_stageC.sh        # status
squeue -u $USER               # queue
cat outputs_stageC/state/STOP # why it stopped
bash package_results.sh       # zip the results
```
