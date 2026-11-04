# Crop-JEPA Stage B on the SPIT GPU cluster

## Run it (3 commands)
```bash
# 1. on your laptop: upload the folder (replace USER and HOST with your cluster login)
scp -r crop_jepa_cluster USER@HOST:~/
# 2. on the cluster
cd ~/crop_jepa_cluster
bash run_all.sh
# 3. whenever you want to look
bash monitor.sh
```
`run_all.sh` submits a CPU job (environment + dataset download) and then the GPU job (smoke test -> Gate 0 pilot ->
full grid -> statistics/figures/tables). You can log out; nothing needs you.

## Where everything is
All results: `outputs_stageB/`   (start with `outputs_stageB/results_bundle.zip`)
- `runs/`          one folder per run (config, train log, checkpoints, predictions, eval.json)
- `stats/`         gates, pre-registration, benchmark, all statistics (CSV/JSON)
- `figures/`       all plots      - `paper_tables/`  CSV + LaTeX tables
- `logs/`          pipeline.log (the readable story), phase logs, SLURM .out/.err
- `smoke/`         output of the smoke test (generated data; meaningless numbers, pipeline check only)
- `state/`         progress markers (DONE / STOP / ...)
Input data (not an output): `lucas_data/`.  Python environment: `venv/`.

## If something stops
Run `bash run_all.sh` again. Finished runs are skipped, unfinished runs resume from their last checkpoint (at most 5 epochs lost).

## Differences from the notebook (scientific code is identical)
1. Configuration comes from `config.env` instead of the first cell.
2. Headless: matplotlib Agg backend, tables printed as text.
3. Pilot and grid workers stop before the reporting cells (`CJ_SKIP_REPORT=1`); one final pass produces the report.
4. The grid is split over `NUM_SHARDS` worker processes sharing the single GPU (the notebook's own SHARD_ID/NUM_SHARDS feature).
