#!/bin/bash
generated_dir="./generated/"
ldload=/home/drc/build-frontend/build/frontend-py.sh

source /home/drc/frontend-venv/bin/activate

task_file=$1
if [[ "$task_file" == "" ]]; then
	echo No task list specified
	exit 1
fi

num_tasks=$(<"${task_file}" wc -l)
echo Submitting $num_tasks tasks
time_tag=$(date +%y%m%d-%H%M%S)
log_dir="logs/${time_tag}"
mkdir -p "${log_dir}"
echo Log directory: $log_dir
sbatch --array="1-$num_tasks" ./run_single_test.sh "$log_dir" "$task_file"
