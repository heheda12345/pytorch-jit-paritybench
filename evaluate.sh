#!/bin/bash
generated_dir="./generated/"
ldload=/home/drc/build-frontend/build/frontend-py.sh

source /home/drc/frontend-venv/bin/activate

for file in "${generated_dir}"test_hirofumi0810_neural_sp.py; do
    echo "evaluate ${file}"
    # echo "evaluate ${file}" >> eval_out
    srun --exclusive=user -A big -p Mix --gres=gpu:1 env LD_PRELOAD=/home/drc/build-frontend/build/ldlong.v3.9.12.so python3 main.py --compile_mode=sys --evaluate-one "$file" --no-fork > temp
    # baseline
    # srun --exclusive --gres=gpu:v100:1 python main.py --evaluate-one "$file" --no-fork > temp0 2>> result2.log
done