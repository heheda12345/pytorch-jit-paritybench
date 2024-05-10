#!/bin/bash
generated_dir="./generated/"
ldload=/home/drc/build-frontend/build/frontend-py.sh

source /home/drc/frontend-venv/bin/activate

for file in "${generated_dir}"test_1adrianb_face_alignment.py; do
    echo "evaluate ${file}"
    # echo "evaluate ${file}" >> eval_out
    srun -A public -p octave --gres=gpu:a100:1 env LD_PRELOAD=/home/drc/build-frontend/build/ldlong.v3.9.12.so python3 main.py --compile_mode=dynamo --evaluate-one "$file" --no-fork > temp
    # srun -A public -p ja --gres=gpu:v100:1 env LD_PRELOAD=/home/drc/build-frontend/build/ldlong.v3.9.12.so python3 main.py --compile_mode=sys --evaluate-one "$file" --no-fork > temp
    # baseline
    # srun --exclusive --gres=gpu:v100:1 python main.py --evaluate-one "$file" --no-fork > temp0 2>> result2.log
done