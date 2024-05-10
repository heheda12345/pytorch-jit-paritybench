#!/bin/bash


#SBATCH -A public
#SBATCH -p octave
#SBATCH --gres=gpu:a100:1
#SBATCH -o /dev/null

ID=${SLURM_ARRAY_TASK_ID}
TASK_FILE="$2"
LOG_PREFIX="$1"
mode="$3"
echo mode $mode
test_file=$(sed "${ID}q;d" "$TASK_FILE")
test_name=$(basename ${test_file})


(
echo Running test $test_file...
echo ======================================= Start time: $(date)
env LD_PRELOAD=/home/drc/build-frontend/build/ldlong.v3.9.12.so python3 main.py --compile_mode="$mode" --evaluate-one "${test_file}" --no-fork
echo ======================================= End time: $(date)
) 2> "$LOG_PREFIX/$test_name.log"