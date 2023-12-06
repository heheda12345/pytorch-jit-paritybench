#!/bin/bash
TIME_TAG=$(date +%Y%m%d-%H%M%S)
srun --exclusive python3 main.py --compile_mode dynamo > dynamo_${TIME_TAG}.out 2>dynamo_${TIME_TAG}.err