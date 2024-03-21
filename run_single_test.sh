#!/bin/bash

#SBATCH --exclusive=user
#SBATCH -A big
#SBATCH -p Mix
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null

ID=${SLURM_ARRAY_TASK_ID}
TASK_FILE="$2"
LOG_PREFIX="$1"

test_file=$(sed "${ID}q;d" "$TASK_FILE")
test_name=$(basename ${test_file})


(
echo Running test $test_file...
echo ======================================= Start time: $(date)
env LD_PRELOAD=/home/drc/build-frontend/build/ldlong.v3.9.12.so python3 main.py --compile_mode=sys --evaluate-one "${test_file}" --no-fork
echo ======================================= End time: $(date)
) 2> "$LOG_PREFIX/$test_name.log"