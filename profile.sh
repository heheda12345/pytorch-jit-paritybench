#!/bin/bash

log=$1
execution_mode=$2

echo profiling "$log" with "$execution_mode" mode
cat "$log"/* > .temp_err
python statistic.py $execution_mode .temp_err "result_$execution_mode"