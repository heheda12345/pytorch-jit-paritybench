A test suite to measure TorchScript parity with PyTorch on many `nn.Module`s
crawled from popular GitHub projects.


###  Running ParityBench

- [Install conda] with python>=3.8
and create/activate a [conda environment]

- Install requirements:
```
conda install pip
pip install -r requirements.txt
conda install pytorch torchvision cpuonly -c pytorch-nightly
```

- Run `python main.py`, you should see an output like:
```
TorchScript ParityBench:
          total  passing  score
projects   1172      346  29.5%
tests      8292     4734  57.1%
```
A file `errors.csv` is generated containing the top error messages and example
`generated/*` files to reproduce those errors.

[Install conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
[conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


### Regenerate ParityBench

*WARNING*: this will download 10+ gigabytes of code from crawling github and
take days to complete.  It is likely not necessary for you to do this.
```
python main.py --download
python main.py --generate-all
```

### Download, generate, evaluate 
You can limit number of github projects to download for testing and running on a smaller set of github repos
```
python main.py --download --download-dir <folder path> --limit 10
```
You can generate tests for one project folder `-g`. This will extract nn modules from that project and generate a test script `--tests-dir`
```
python main.py -g <folder path> --tests-dir <folder path>
```
You can evaluate one generated test script `-e` and try export the module to onnx `--onnxdir` 
```
python main.py -e <test.py file> --onnxdir <folder path>
```
You can evaluate using different compile mode, e.g, `dynamo`(default) or `torchscript`.
```
python main.py -e <test.py file> --compile_mode dynamo
```
You can evaluate using different dynamo backends provided in `torch._dynamo`, please refer `torch._dynamo.list_backends()`.
```
python main.py -e <test.py file> --backend eager
```
You can evaluate using `cuda`(default) or `cpu`.
```
python main.py -e <test.py file> --device cuda
```

### Running with FROTEND

you can use new compile mode ```sys``` to generate more full graphs in this benchmark!

**for single model evaluation**

``` 
# check and update your configures in this script: preload directory, GPU command and so on
./evaluate.sh
```

**for all models in this repo**

```
# check and update configures in your machine
# NOTE: we use sbatch to run multiple jobs
./evaluate-all.sh test-list.txt
```

**you can use**
```
cat logs/your_generated_directory/* > profiling_result
```
and integrate all model results

**you can use**
```
python statistic.py your_generated_profiling_result_directory stats
```
to see what scores our system can reach, the ```stats``` file includes details of failed models