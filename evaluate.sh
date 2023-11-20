#!/bin/bash
LD_PRELOAD=/home/drc/frontend/build/ldlong.v3.9.12.so
generated_dir="./generated/"

for file in "${generated_dir}"test_*.py; do
    echo "evaluate ${file}"
    # echo "evaluate ${file}" >> eval_out
    srun --exclusive python main.py --evaluate-one "$file" --no-fork > temp 2>> error_output
done
