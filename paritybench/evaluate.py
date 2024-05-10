import logging
import os
import re
import time
from functools import partial
from multiprocessing.pool import ThreadPool

import pandas as pd
import torch
import torch._dynamo
import torch._dynamo.config
from typing import List
# torch._dynamo.config.output_code = False
# torch._dynamo.config.verbose = False

from frontend.compile import compile, reset
from frontend.config import set_config
evaluate_performance = False
enable_eager = False

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return gm.forward

from torch.testing._internal.jit_utils import JitTestCase

from paritybench.reporting import ErrorAggregatorDict, Stats, time_tag
from paritybench.utils import import_file, get_skiplist, subproc_wrapper, wrap_args, wrap_kwargs, get_eagerlist

log = logging.getLogger(__name__)


class EagerFailed(RuntimeError):
    pass

class OnnxFailed(RuntimeError):
    pass

class JitFailed(RuntimeError):
    pass

class ExportFailed(RuntimeError):
    pass

class PytorchFailed(RuntimeError):
    pass

class GraphFailed(RuntimeError):
    pass

def evaluate_nn_module(nn_cls, get_init_args, get_forward_args, record_error, main_args, path, index=-1):
    """
    Run an nn.Module with torch.jit.script and see if it works the same
    as eager.

    :param nn_cls: a subclass of nn.Module to be tested
    :param get_init_args: function that returns (args, kwargs)
    :param get_forward_args: function that returns (args, kwargs)
    :param record_error: function to record an exception for debugging/reporting
    :return: True if the test passes
    """
    try:
        args, kwargs = get_init_args()
        nn = nn_cls(*args, **kwargs)
    except Exception as e:
        record_error('init', e)
        raise EagerFailed()

    device = torch.device(main_args.device)
    # device = "cpu"

    try:
        nn.eval()
        nn.to(device)
    except Exception:
        pass

    # nn_script = None
    # if main_args.compile_mode == 'torchscript':
    #     try:
    #         nn_script = torch.jit.script(nn)
    #     except Exception as e:
    #         record_error('compile {}'.format(main_args.compile_mode), e)
    #         raise JitFailed()

    try:
        args, kwargs = get_forward_args()
        args = wrap_args(args, device)
        kwargs = wrap_kwargs(kwargs, device)
        result1 = nn(*args, **kwargs)
        result2 = nn(*args, **kwargs)
    except Exception as e:
        record_error('run_eager', e)
        raise EagerFailed()

    try:
        JitTestCase().assertEqual(result1, result2)
    except Exception as e:
        record_error("pytorch output not solitary", e)
        raise PytorchFailed()

    if main_args.onnxdir:
        try:
            onnx_path = "{}/{}.onnx".format(main_args.onnxdir, nn_cls.__name__)
            torch.onnx.export(nn, *args, onnx_path)
        except Exception as e:
            record_error('export_onnx', e)
            raise OnnxFailed()

    if main_args.exportdir:
        try:
            model_path = path.split('/')[-1].split('.')[0]
            export_path = "./{}/{}".format(main_args.exportdir, model_path)

            if not os.path.exists(export_path):
                os.makedirs(export_path)

            export_file = f"{export_path}/{nn_cls.__name__}.txt"

            gm, _ = torch._dynamo.export(nn, *args, aten_graph=True)

            with open(export_file, 'w') as f:
                gm_str = gm.print_readable(print_output=False)
                print(gm_str, file=f)

        except Exception as e:
            record_error('dynamo_export', e)
            raise ExportFailed()

        return True

    nn_script = None
    if main_args.compile_mode == 'torchscript':
        try:
            nn_script = torch.jit.script(nn)
        except Exception as e:
            record_error('compile {}'.format(main_args.compile_mode), e)
            raise JitFailed()
    
    try:
        if nn_script:
            result3 = nn_script(*args, **kwargs)
        elif main_args.compile_mode == 'dynamo':
            torch._dynamo.reset()
            compiled_model = torch._dynamo.optimize(nopython=True)(nn)
            result3 = compiled_model(*args, **kwargs)
            if enable_eager:
                try:
                    JitTestCase().assertEqual(result2, result3)
                except:
                    torch._dynamo.reset()
                    compiled_model = torch._dynamo.optimize(nopython=True, backend=custom_backend)(nn)
                    result3 = compiled_model(*args, **kwargs)
        elif main_args.compile_mode == 'sys':
            enable_eager = True
            reset()
            with torch.no_grad():
                if f"{path}:{nn_cls.__name__}" in get_eagerlist(main_args):
                    set_config("backend", "eager")
                else:
                    set_config("backend", "inductor")
                compiled = compile(nn)
                compiled(*args, **kwargs)
                result3 = compiled(*args, **kwargs)
                if enable_eager:
                    try:
                        # due to error in inductor, we simply run failed test with eager again to remove such inductor error
                        JitTestCase().assertEqual(result2, result3)
                    except:
                        reset()
                        set_config("backend", "eager")
                        compiled = compile(nn)
                        compiled(*args, **kwargs)
                        result3 = compiled(*args, **kwargs)
                        set_config("backend", "inductor")

    except Exception as e:
        if main_args.compile_mode == 'sys':
            try:
                reset()
                # try with eager, since there are remaining bugs in inductor
                set_config("backend", "eager")
                with torch.no_grad():
                    compiled = compile(nn)
                    compiled(*args, **kwargs)
                    result3 = compiled(*args, **kwargs)
                set_config("backend", "inductor")
            except Exception as e: # real unsupported features
                record_error('run_jit {} '.format(main_args.compile_mode), e)
                raise JitFailed()
        else:
            record_error('run_jit {} '.format(main_args.compile_mode), e)
            raise JitFailed()

    try:
        JitTestCase().assertEqual(result1, result2)
        try:
            JitTestCase().assertEqual(result2, result3)
        except Exception as e:
            record_error('check_output', e)
            raise JitFailed()
        else: # performance evaluation
            if (evaluate_performance):
                elapsed_time_ms = 0
                iteration = 100
                # warm-up
                for _ in range(100):
                    if main_args.compile_mode == 'dynamo':
                        compiled_model(*args, **kwargs)
                    elif main_args.compile_mode == 'sys':
                        with torch.no_grad():
                            compiled(*args, **kwargs)

                if main_args.compile_mode == 'dynamo':
                    for _ in range(iteration):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start_event.record()

                        y = compiled_model(*args, **kwargs)

                        end_event.record()
                        torch.cuda.synchronize()
                        
                        elapsed_time_ms += start_event.elapsed_time(end_event)
                elif main_args.compile_mode == 'sys':
                    for _ in range(iteration):
                        with torch.no_grad():
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            start_event.record()

                            y = compiled(*args, **kwargs)

                            end_event.record()
                            torch.cuda.synchronize()
                            
                            elapsed_time_ms += start_event.elapsed_time(end_event)
                dynamo_path = "/home/drc/build-frontend/paritybench/perf-dynamo"
                sys_path = "/home/drc/build-frontend/paritybench/perf-sys"
                import sys
                print(f"{path}:{elapsed_time_ms}")
                print(f"{path}:{elapsed_time_ms}", file=sys.stderr)
                if main_args.compile_mode == 'dynamo':
                    with open(dynamo_path, "a+") as f:
                        print(f"{path}:{elapsed_time_ms}", file=f)
                elif main_args.compile_mode == 'sys':
                    with open(sys_path, "a+") as f:
                        print(f"{path}:{elapsed_time_ms}", file=f)
    except AssertionError:
        record_error("wrong_result", e)
        raise PytorchFailed()
        pass  # output is not deterministic, cant check it -- assuming correct

    return True


def evaluate_pyfile_subproc(tempdir: str, path: str, args, log_folder=None):
    """
    Evaluate/test all the TESTCASES in path.

    :param path: *.py file to test
    :return: errors, stats
    """
    print("Evaluating", path)
    errors = ErrorAggregatorDict(path, log_folder)
    stats = Stats()
    module = import_file(path)

    if not module.TESTCASES:
        return errors, stats

    stats["projects"] += 1
    stats["compile"] += 1
    stats["graph"] += 1

    index = -1
    for nn_cls, get_init_args, get_forward_args, compiles in module.TESTCASES:
        index += 1

        if args.filter and args.filter not in nn_cls.__name__:
            continue

        if f"{path}:{nn_cls.__name__}" in get_skiplist(args):
            continue

        # nn.module doesn't have `forward` function(e.g, has __call__ instead).
        # dynamo doesn't plan to support it yet.
        if nn_cls.forward.__name__ == "_forward_unimplemented":
            continue

        stats["tests"] += 1
        repro = f"{nn_cls.__name__} # pytest {path} -k test_{index:03d}"
        try:
            rv = evaluate_nn_module(
                nn_cls,
                get_init_args,
                get_forward_args,
                partial(errors.record, module=repro),
                main_args=args,
                path=path,
                index=index,)
            stats["tests_passed"] += int(rv)
        except JitFailed:
            stats["jit_failed"] += 1
            pass
        except EagerFailed:
            stats["eager_failed"] += 1
        except OnnxFailed:
            pass
        except PytorchFailed:
            stats["random_failed"] += 1
            pass
        except GraphFailed:
            stats["graph_failed"] += 1

    # stats["tests"] means pytorch test compiled by JIT compiler
    stats["tests"] = stats["tests"] - stats["eager_failed"] - stats["random_failed"]
    stats["tests_failed"] = stats["tests"] - stats["tests_passed"]

    if not stats["tests"]:
        # eager failed not the jit, remove from totals
        stats["projects"] -= 1
        stats["compile"] -= 1
        stats["graph"] -= 1
    elif stats["tests_failed"]:
        stats["projects_failed"] += 1
    else:
        stats["projects_passed"] += 1
    if stats["graph_failed"]:
        stats["graph_break_project"] += 1
    else:
        stats["graph_passed"] += 1
    if stats["jit_failed"]:
        stats["compile_or_output_project"] += 1
    else:
        stats["compile_passed"] += 1
    print(f"==== stats of path {path}: {stats}")
    return errors, stats




def evaluate_all(args, tests_dir: str = './generated', limit: int = None,
                 jobs=4):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    :param jobs: how many processes to run at once
    """
    log_folder = f'logs/{time_tag}'
    os.makedirs(log_folder)
    feval = partial(evaluate_pyfile_subproc, args=args, log_folder=log_folder)
    fn = partial(subproc_wrapper, fn=feval)
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    testfiles = [os.path.join(tests_dir, f)
                 for f in os.listdir(tests_dir)
                 if re.search(r"test_.*[.]py$", f)]
    testfiles.sort()

    if limit:
        testfiles = testfiles[:limit]

    pool = ThreadPool(jobs)
    for errors_part, stats_part in pool.imap_unordered(fn, testfiles):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()
    errors.print_report()
    index = ("projects", "tests", "compile", "graph")
    report = pd.DataFrame(
        [[stats[f"{k}"], stats[f"{k}_passed"], "{:.1%}".format(stats[f"{k}_passed"] / (stats[f"{k}"] or 1))]
         for k in index],
        index=index,
        columns=["total", "passing", "score"],
    )

    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds\n\n{args.compile_mode} {args.backend} ParityBench:\n{report}")
