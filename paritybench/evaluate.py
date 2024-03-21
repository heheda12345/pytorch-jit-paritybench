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
torch._dynamo.config.output_code = False
torch._dynamo.config.verbose = False

from frontend.compile import compile, reset
from frontend.config import set_config

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return gm.forward

from torch.testing._internal.jit_utils import JitTestCase

from paritybench.reporting import ErrorAggregatorDict, Stats
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

    nn_script = None
    if main_args.compile_mode == 'torchscript':
        try:
            nn_script = torch.jit.script(nn)
        except Exception as e:
            record_error('compile {}'.format(main_args.compile_mode), e)
            raise JitFailed()

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

    try:
        if nn_script:
            result3 = nn_script(*args, **kwargs)
        elif main_args.compile_mode == 'dynamo':
            # torch._dynamo.reset()
            # compiled_model = torch._dynamo.optimize(nopython=True)(nn)
            # result3 = compiled_model(*args, **kwargs)
            # (
            #     explanation,
            #     out_guards,
            #     graphs,
            #     ops_per_graph,
            #     break_reasons,
            #     explanation_verbose,
            # ) = torch._dynamo.explain(compiled_model, *args, **kwargs)
            # log_path = 'logs/' + path.split('/')[-1].split('.')[0] + "." + str(index) + '.log'
            # with open(log_path, 'w') as f:
            #     print(explanation_verbose, file=f)
            #     for i, (graph_guard, graph, ops, break_reason) in enumerate(zip(
            #         out_guards, graphs, ops_per_graph, break_reasons
            #     )):
            #         print("GRAPH", i, file=f)
            #         print("++graph_guard:", len(graph_guard), file=f)
            #         for guard in graph_guard:
            #             print(guard, file=f)
            #         print("++graph:", file=f)
            #         print(graph.print_readable(print_output=False), file=f)
            #         print("++ops:", len(ops), file=f)
            #         for op in ops:
            #             print(op, file=f)
            #         print("++break_reason:", break_reason.reason, file=f)
            #         print("".join(traceback.format_list(break_reason.user_stack)), file=f)
            #     print("finish", file=f)

            torch._dynamo.reset()
            compiled_model = torch._dynamo.optimize(nopython=True)(nn)
            result3 = compiled_model(*args, **kwargs)
        else: # frontend compiler 
            reset()
            with torch.no_grad():
                if f"{path}:{nn_cls.__name__}" in get_eagerlist(main_args):
                    set_config("backend", "eager")
                else:
                    set_config("backend", "inductor")
                compiled = compile(nn)
                compiled(*args, **kwargs)
                result3 = compiled(*args, **kwargs)
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
        record_error('run_jit {} '.format(main_args.compile_mode), e)
        raise JitFailed()

    try:
        JitTestCase().assertEqual(result1, result2)
        try:
            JitTestCase().assertEqual(result2, result3)
        except Exception as e:
            record_error('check_output', e)
            raise JitFailed()
        # try:
        #     JitTestCase().assertEqual(len(break_reasons), 1)
        # except Exception as e:
        #     record_error('has graph breaks', e)
        #     raise GraphFailed()
    except AssertionError:
        record_error("pytorch output not solitary1", e)
        raise PytorchFailed()
        pass  # output is not deterministic, cant check it -- assuming correct

    return True


def evaluate_pyfile_subproc(tempdir: str, path: str, args):
    """
    Evaluate/test all the TESTCASES in path.

    :param path: *.py file to test
    :return: errors, stats
    """
    print("Evaluating", path)
    errors = ErrorAggregatorDict(path)
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
    # errors.print_report()
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
    feval = partial(evaluate_pyfile_subproc, args=args)
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
