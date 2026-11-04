#!/bin/bash
# Submits stageC_job.sbatch with the GPU / CPU / memory / time values of config_stageC.env
# (command-line options override the #SBATCH defaults in the job file). Prints the job id.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
source ./config_stageC.env
sbatch --parsable --partition="$PARTITION" --gres=gpu:"$NUM_GPUS" --cpus-per-task="$CPUS_PER_TASK" --mem="${MEM_GB}G" \
  --time="$WALL_TIME" "$@" stageC_job.sbatch
