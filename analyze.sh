#!/bin/bash

cat logs/dynamo/* > .dynamo_temp_err
cat logs/torchscript/* > .script_temp_err
cat logs/sys/* > .sys_temp_err

commands=(
    "python statistic.py sys .sys_temp_err result_sys"
    "python statistic.py dynamo .dynamo_temp_err result_dynamo"
    "python statistic.py torchscript .script_temp_err result_torchscript"
)

header=false

for cmd in "${commands[@]}"; do
  output=$($cmd)
  mode=$(echo "$cmd" | awk '{print $3}')
  if [ "$mode" == "sys" ]; then
    echo "$output" | head -n 12
    echo ""
    printf "%-12s %-10s %-10s %-10s\n" "mode" "total" "Failed cases" " Fail rate"
    header=true
  fi

  line=$(echo "$output" | grep -P "^models\s+(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+%)$" | awk '{print $2, $3, $4}')
  if [ "$header" = false ]; then
    printf "%-12s %-12s %-15s %-10s\n" "mode" "total" "Failed cases" "Fail rate"
    header=true
  fi
  

  printf "%-12s %-10s %-10s %-10s\n" "$mode" $line
done