#!/bin/bash

task_file=$1
execution_mode=$2
if [[ "$task_file" == "" ]]; then
	echo No task list specified
	exit 1
fi

num_tasks=$(<"${task_file}" wc -l)
echo Running with $execution_mode mode
echo Submitting $num_tasks tasks

log_dir="logs/${execution_mode}"
mkdir -p "${log_dir}"
echo Log directory: $log_dir
sbatch --array="1-$num_tasks" ./run_single_test.sh "$log_dir" "$task_file" "$execution_mode"
