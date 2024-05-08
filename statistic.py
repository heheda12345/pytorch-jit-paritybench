import re
import pandas as pd
import itertools
import sys

file_pattern = r"INFO \[utils\.py:\d+\] Running (.+?\.py)"

pattern = r"INFO \[main\.py:\d+\] Stats: \[\('total', (\d+)\), \('init_ok', (\d+)\), \('deduced_args_ok', (\d+)\), \('jit_compiles', (\d+)\), \('projects', (\d+)\), \('compile', (\d+)\), \('graph', (\d+)\), \('projects_passed', (\d+)\), \('projects_failed', (\d+)\), \('graph_passed', (\d+)\), \('compile_passed', (\d+)\), \('tests', (\d+)\), \('tests_passed', (\d+)\), \('tests_failed', (\d+)\), \('random_failed', (\d+)\), \('graph_failed', (\d+)\), \('jit_failed', (\d+)\), \('graph_break_project', (\d+)\), \('compile_or_output_project', (\d+)\)\]"

eager_pattern = r"INFO \[main\.py:\d+\] Stats: \[\('total', (\d+)\), \('init_ok', (\d+)\), \('deduced_args_ok', (\d+)\), \('jit_compiles', (\d+)\), \('projects', (\d+)\), \('compile', (\d+)\), \('graph', (\d+)\), \('projects_passed', (\d+)\), \('projects_failed', (\d+)\), \('graph_passed', (\d+)\), \('compile_passed', (\d+)\), \('tests', (\d+)\), \('tests_passed', (\d+)\), \('tests_failed', (\d+)\), \('random_failed', (\d+)\), \('graph_failed', (\d+)\), \('jit_failed', (\d+)\), \('graph_break_project', (\d+)\), \('compile_or_output_project', (\d+)\), \('eager_failed', (\d+)\)\]"

bug_pattern = r"ERROR \[reporting\.py:\d+\] run_jit sys  error from (.*\.py):(.*?)-k (test_\d+)"

output_pattern = r"ERROR \[reporting\.py:\d+\] check_output error from (.*\.py):(.*?)-k (test_\d+)"

models = []
bug_info = {}
output_info = {}
total_failed = set()

class Model():
    def __init__(self, filepath:str) -> None:
        self.filepath = filepath
        self.profiling = {}

def collector(match):
    stats_data = {
        # 'total': int(match.group(1)),
        # 'init_ok': int(match.group(2)),
        # 'deduced_args_ok': int(match.group(3)),
        # 'jit_compiles': int(match.group(4)),
        'projects': int(match.group(5)),
        # 'projects_passed': int(match.group(6)),
        'projects_failed': int(match.group(9)),
        'tests': int(match.group(12)),
        'tests_passed': int(match.group(13)),
        'tests_failed': int(match.group(14)),
        'random_failed': int(match.group(15))
    }
    return stats_data

def bug_collector(match):
    file_name = match.group(1)
    test_info = match.group(3)
    if file_name in bug_info:
        bug_info[file_name].append(test_info)
    else:
        bug_info[file_name] = [test_info]
    return file_name

def output_collector(match):
    file_name = match.group(1)
    test_info = match.group(3)
    if file_name in output_info:
        output_info[file_name].append(test_info)
    else:
        output_info[file_name] = [test_info]
    return file_name

def collect(profiling_file_path):
    with open(profiling_file_path, 'r') as file:
        model = None
        for i, content in enumerate(file):
            file_match = re.match(file_pattern, content)
            std_match = re.match(pattern, content)
            eager_match = re.match(eager_pattern, content)
            bug_match = re.match(bug_pattern, content)
            output_match = re.match(output_pattern, content)
        
            filename = ""
            if file_match:
                model = Model(file_match.group(1))
            if std_match and model is not None:
                model.profiling = collector(std_match)
            if eager_match and model is not None:
                model.profiling = collector(eager_match)
            if bug_match and model is not None:
                filename = bug_collector(bug_match)
                assert filename in model.filepath
            if output_match and model is not None:
                filename = output_collector(output_match)
                assert filename in model.filepath
            if model is not None and model not in models:
                models.append(model)
    for project in itertools.chain(bug_info.keys(), output_info.keys()):
        total_failed.add(project)

def process(output_file_path):
    stats = {}
    passed_model = 0
    no_profiling = 0
    no_test = 0
    dynamic_models = 128
    for i in models:
        # if i.filepath not in total_failed:
        #     passed_model += 1
        if len(i.profiling) == 0:
            no_profiling += 1
        if len(i.profiling) != 0 and i.profiling['tests'] == 0:
            no_test += 1
    # print("passed model", passed_model)
    print("overall models", len(models))
    print("   untested models:", no_test + no_profiling)
    print("      no tests: ", no_test)
    print("      no profiling: ", no_profiling)
    print()
    print("   dynamic models: ", dynamic_models)
    tested_models = len(models) - no_profiling - no_test - dynamic_models
    stats["models"], stats["passing"], stats["output"] = tested_models, tested_models, tested_models
    stats["models_info"] = len(total_failed) - dynamic_models
    stats["passing_info"] = len(bug_info) - dynamic_models
    stats["output_info"] = len(output_info)
    print()

    with open(output_file_path, 'w') as output_file:
        output_file.write('----------------------------------------------------------------' + '\n')
        print(f"bug failed: {len(bug_info)}, as below:", file=output_file)
        for bug, testcase in bug_info.items():
            output_file.write(str(bug) +':' + str(testcase)+ '\n')
        output_file.write('----------------------------------------------------------------' + '\n')
        print(f"output failed: {len(output_info)}, as below:", file=output_file)
        for bug, testcase in output_info.items():
            output_file.write(str(bug) +':' + str(testcase)+ '\n')
        output_file.write('----------------------------------------------------------------' + '\n')
        print(f'total failed:{len(total_failed)}\n', file=output_file)
        for project in total_failed:
            output_file.write(str(project) + '\n')
        output_file.write('----------------------------------------------------------------' + '\n')
        index = index = ("models", "passing", "output")
        report = pd.DataFrame(
            [[stats[f"{k}"], (stats[f"{k}"] - stats[f"{k}_info"]), "{:.1%}".format((stats[f"{k}"] - stats[f"{k}_info"]) / (stats[f"{k}"] or 1))]
            for k in index],
            index=index,
            columns=["total", "passed", "score"],
        )
        print(report)
        print(report, file=output_file)

def main():
    #TODO: integrate with evaluate-all.sh
    if len(sys.argv) != 3:
        print("Usage: ")
        print("       python statistic.py profiling_file statistic_result")
        exit(1)
    else:
        profiling_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        # print("collecting...")
        collect(profiling_file_path)
        print("profiling...")
        process(output_file_path)
main()