import os
from paritybench.utils import import_file
import torch
from torch.nn.parallel import DistributedDataParallel

def should_test_cls(obj, module):
    if not isinstance(obj, type): # check if is class
        return False
    if not issubclass(obj, torch.nn.Module): # check if is torch module
        return False
    if issubclass(obj, DistributedDataParallel):
        return False
    return obj.__module__ == module.__name__


def check_test(path):
    try:
        module = import_file(path)
        all_testcases = set(x[0] for x in module.TESTCASES)
        n_fail = 0
        n_case = 0
        for name, value in module.__dict__.items():
            if should_test_cls(value, module):
                if value not in all_testcases:
                    print("x", name, "not in", path)
                    n_fail += 1
                else:
                    print("v", name, "in", path)
                    n_case += 1
        return (path, n_case, n_fail)
    except Exception:
        return None

def gen_accept_rate():
    all_test = []
    for i, path in enumerate(os.listdir("generated")):
        if "test" not in path: continue
        out = check_test(os.path.join("generated", path))
        if out is not None:
            all_test.append(out)
        print(i, "finish")
        # if i == 10: break

    all_test.sort(key = lambda x: x[1] / (x[1] + x[2]) if x[1]+x[2] > 0 else 0)

    for i, x in enumerate(all_test):
        print(i, x, x[1] / (x[1] + x[2]) if x[1]+x[2] > 0 else 0)

    with open("accept_rate", "w") as f:
        for x in all_test:
            f.write(f"{x[0].split('/')[-1][:-3]} {x[1]} {x[1] + x[2]} {x[1] / (x[1] + x[2]) if x[1]+x[2] > 0 else 0}\n")


def analyze():
    import matplotlib.pyplot as plt
    all_full_graph = set()
    with open("pt20full_graph") as f:
        for s in f.readlines():
            all_full_graph.add(s.strip())
    case_rate = []
    full_graph_rate = []
    total_is_full = 0
    total_hastest = 0
    with open("accept_rate") as f:
        all_log = []
        for _, s in enumerate(f.readlines()):
            test_name, n_case, n_model, rate = s.strip().split()
            n_case = int(n_case)
            n_model = int(n_model)
            rate = float(rate)
            if n_case == 0: continue
            all_log.append((test_name, n_case, n_model, rate))
    for test_name, n_case, n_model, rate in reversed(all_log):
        total_hastest += 1
        is_full = test_name in all_full_graph
        case_rate.append(rate)
        total_is_full += is_full
        full_graph_rate.append(total_is_full / total_hastest)
        print(test_name, rate, is_full, n_case, n_model)
    # plt.scatter(case_rate, full_graph_rate)
    # plt.xlabel("is testcase rate")
    # plt.ylabel("is fullgraph rate (accumulated)")
    # plt.savefig("case_rate-full_graph.png")
    num_bucket = 10
    buckets = [[0,0] for _ in range(num_bucket + 1)]
    for test_name, n_case, n_model, rate in reversed(all_log):
        idx = int(rate * num_bucket)
        buckets[idx][0] += 1
        buckets[idx][1] += int(test_name in all_full_graph)
    plt.bar([x for x in range(num_bucket + 1)], [b/a for a, b in buckets]) 
    plt.savefig("case_rate-full_graph-bar.png")
    print([b[0] for b in buckets])




# gen_accept_rate()
analyze()