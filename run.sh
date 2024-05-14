#!/bin/bash

echo "running torchscript"
./evaluate-all.sh test_list.txt torchscript
echo "running dynamo"
./evaluate-all.sh test_list.txt dynamo
echo "running sys"
./evaluate-all.sh test_list.txt sys