# Crop-JEPA follow-up: the 2×2 confirmation run (optional, post hoc)

## What it tests, in one paragraph
Stage C showed that veg_pos's small Stage B advantage disappears under the Stage C protocol. That protocol changed **two things at once**: it gave every arm the same 12 context tokens, and it trained 3× longer (26k instead of 9k steps). This follow-up runs the two missing combinations, so each factor can be tested separately:

| | 9,030 steps | 26,058 steps |
|---|---|---|
| **no cap** (Stage B protocol) | Stage C **bridge** (already done) | **uncap26k** (new) |
| **cap K = 12** (Stage C protocol) | **cap9k** (new) | Stage C **core**, seeds 100–109 (already done) |

Setup: ijepa and veg_pos only, seeds 100–109, split A, ViT-Tiny. That is 2 arms × 10 seeds × 2 cells = **40 runs**.
- **Estimated time:** ~2.5 h on 2 GPUs (6 workers), ~4.5 h on 1 GPU; 12 h requested. Set `NUM_GPUS=1` in `followup_common.sh` if only one GPU is free (then also `CPUS_PER_TASK=32 MEM_GB=64`).
- **Order:** cap9k runs first. On its own, combined with the bridge, it already answers "does the cap alone remove the effect at the Stage B length?"

**Status:** this is an *exploratory, post-hoc* follow-up, designed after the Stage C results were known. It is still frozen before its first run: `outputs_followup/followup_plan.json` + `.sha256` hold the plan, the analysis rules, the follow-up code hash and the Stage C pre-registration hash. In the paper, call it "a post-hoc follow-up, frozen before running", **not** pre-registered.

## Safety
- New, separate folder: `~/crop_jepa_followup`. Nothing in `~/crop_jepa_cluster` (Stage B) or `~/crop_jepa_stageC` (Stage C) is written.
- It only **reads**:
  - the Stage B image cache and split A;
  - Stage C's Phase-0 decisions (cap 12);
  - the frozen Stage C pre-registration and its confirmation record (both verified unchanged);
  - Stage C core and bridge results (for the 2×2 table);
  - the Stage C Python environment (`~/crop_jepa_stageC/venv`; reused, no new install).
- `crop_jepa_followup.py` is generated from the Stage C script by `make_followup_script.py`. It applies 8 exact, checked text patches: the two new phases, their freeze record and report labels. Model, masking, training, evaluation and statistics code is identical.

## Commands

### 1. Upload (laptop, PowerShell)
```powershell
ssh <username>@cluster.gpu.spit.ac.in "ls -d ~/crop_jepa_followup 2>/dev/null || echo 'no follow-up folder yet - OK'"
scp -r "C:\Users\kalhan\Desktop\crop_jepa_cluster\crop_jepa_followup" <username>@cluster.gpu.spit.ac.in:~/
```
If the first command prints a path instead of `OK`, the folder already exists. Don't upload over it; ask first.

### 2. Check and submit (cluster)
```bash
ssh <username>@cluster.gpu.spit.ac.in
cd ~/crop_jepa_followup
sed -i 's/\r$//' *.sh *.sbatch *.py
sha256sum crop_jepa_followup.py          # bc5adf39e6c3f74aa9408edb2275e6de5d91398c95ba631bf74309e5d6066ca5
ls ~/crop_jepa_stageC/outputs_stageC/state/DONE ~/crop_jepa_stageC/venv/bin/activate   # both must exist
squeue -u $USER                          # nothing named cjF yet
bash run_followup.sh
```
Expected output:
```
Stage C pre-registration sha256 (must be c6774dac...): c6774dac49cc7f3d...
Submitted follow-up GPU job NNNN (2 GPU(s), 48 CPUs, 96 GB, 12:00:00).
```
If it prints `MISSING (read-only input): ...`, nothing was submitted. Tell me which file is missing.

### 3. Watch (any time)
```bash
cd ~/crop_jepa_followup
bash monitor_followup.sh                 # queue, state, runs finished per phase (x/20), latest training lines
tail -f outputs_followup/logs/pipeline.log
```
What the job writes:
- `outputs_followup/followup_plan.json` (+ `.sha256`), the moment the cap9k prepare pass starts (before any run);
- `state/cap9k.done`, then `state/uncap26k.done`;
- `state/DONE` and `outputs_followup/FOLLOWUP_2x2_REPORT.md`.

### 4. If it stops
```bash
cat outputs_followup/state/STOP
bash run_followup.sh                     # resumes: finished runs are skipped, unfinished ones continue from checkpoint
```
- If it ran out of time, just run `bash run_followup.sh` again.
- If it stopped with an error, send me the last 40 lines of the log the STOP message names.
- **If time runs out before uncap26k finishes**, you still get a useful partial answer. Run the analysis by hand:
  ```bash
  source ~/crop_jepa_stageC/venv/bin/activate && source ./followup_common.sh && python followup_2x2_report.py
  ```
  It reports whichever cells exist.

### 5. Collect (cluster, then laptop)
```bash
cd ~/crop_jepa_followup
cat outputs_followup/FOLLOWUP_2x2_REPORT.md
bash package_followup.sh                 # followup_results.zip + .sha256
```
```powershell
scp <username>@cluster.gpu.spit.ac.in:~/crop_jepa_followup/followup_results.zip "C:\Users\kalhan\Desktop\crop_jepa_cluster\"
Get-FileHash "C:\Users\kalhan\Desktop\crop_jepa_cluster\followup_results.zip" -Algorithm SHA256
Expand-Archive "C:\Users\kalhan\Desktop\crop_jepa_cluster\followup_results.zip" -DestinationPath "C:\Users\kalhan\Desktop\crop_jepa_cluster\followup_results"
```

## How the answer will be read (fixed before running)
All contrasts are per seed, on d = veg_pos − ijepa (primary endpoint, pp):
- **cap_at_9k** = d(cap9k) − d(bridge): what the cap alone does at the Stage B length.
- **length_uncapped** = d(uncap26k) − d(bridge): what longer training alone does under the Stage B protocol.
- Also reported: cap_at_26k, length_capped, both main effects, the interaction, the same on k-NN, and the H5 row-crop moderation per cell.

Reading rule:
- If cap_at_9k has a CI below 0 and length_uncapped includes 0, **the cap explains it**.
- If length_uncapped has a CI below 0 and cap_at_9k includes 0, **training length explains it**.
- If both are below 0, **both** contribute.
- Otherwise, **neither alone is identified**.

With 10 seeds the results are exploratory: report estimates with CIs, never "significant".
