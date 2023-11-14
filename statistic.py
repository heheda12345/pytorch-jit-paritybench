import re
import pandas as pd
import itertools

pattern = r"INFO \[main\.py:\d+\] Stats: \[\('total', (\d+)\), \('init_ok', (\d+)\), \('deduced_args_ok', (\d+)\), \('jit_compiles', (\d+)\), \('projects', (\d+)\), \('projects_passed', (\d+)\), \('projects_failed', (\d+)\), \('tests', (\d+)\), \('tests_passed', (\d+)\), \('tests_failed', (\d+)\), \('random_failed', (\d+)\)\]"

bug_pattern = r"ERROR \[reporting\.py:\d+\] run_jit dynamo  error from (.*\.py):(.*?)-k (test_\d+)"

output_pattern = r"ERROR \[reporting\.py:\d+\] check_output error from (.*\.py):(.*?)-k (test_\d+)"


output_file_path = 'output_file.txt'

with open('eval_out', 'r') as file:
    content = file.read()

matches = re.finditer(pattern, content)
bug_matches = re.finditer(bug_pattern, content)
output_matches = re.finditer(output_pattern, content)
result = []
bug_info = {}
output_info = {}

for match in matches:
    stats_data = {
        # 'total': int(match.group(1)),
        # 'init_ok': int(match.group(2)),
        # 'deduced_args_ok': int(match.group(3)),
        # 'jit_compiles': int(match.group(4)),
        # 'projects': int(match.group(5)),
        # 'projects_passed': int(match.group(6)),
        'projects_failed': int(match.group(7)),
        'tests': int(match.group(8)),
        'tests_passed': int(match.group(9)),
        'tests_failed': int(match.group(10)),
        'random_failed': int(match.group(11))
    }
    result.append(stats_data)

for bug in bug_matches:
    file_name = bug.group(1)
    test_info = bug.group(3)
    if file_name in bug_info:
        bug_info[file_name].append(test_info)
    else:
        bug_info[file_name] = [test_info]

for output in output_matches:
    file_name = output.group(1)
    test_info = output.group(3)
    if file_name in output_info:
        output_info[file_name].append(test_info)
    else:
        output_info[file_name] = [test_info]

total_failed = set()
for project in itertools.chain(bug_info.keys(), output_info.keys()):
    total_failed.add(project)

stats = {}
stats["projects"], stats["bug"], stats["output"] = 1414, 1414, 1414
stats["projects_info"] = len(total_failed)
stats["bug_info"] = len(bug_info)
stats["output_info"] = len(output_info)

with open(output_file_path, 'w') as output_file:
    for stats_data in result:
        output_file.write(str(stats_data) + '\n')
    output_file.write('----------------------------------------------------------------' + '\n')
    for bug, testcase in bug_info.items():
        output_file.write(str(bug) +':' + str(testcase)+ '\n')
    output_file.write('----------------------------------------------------------------' + '\n')
    for bug, testcase in output_info.items():
        output_file.write(str(bug) +':' + str(testcase)+ '\n')
    output_file.write('----------------------------------------------------------------' + '\n')
    print(f'total failed:{len(total_failed)}\n', file=output_file)
    for project in total_failed:
        output_file.write(str(project) + '\n')
    output_file.write('----------------------------------------------------------------' + '\n')
    index = index = ("projects", "bug", "output")
    report = pd.DataFrame(
        [[stats[f"{k}"], (stats[f"{k}"] - stats[f"{k}_info"]), "{:.1%}".format((stats[f"{k}"] - stats[f"{k}_info"]) / (stats[f"{k}"] or 1))]
         for k in index],
        index=index,
        columns=["total", "passing", "score"],
    )
    print(report, file=output_file)



# df = pd.DataFrame(output_info.items())

# excel_file_path = 'error_data.xlsx'
# df.to_excel(excel_file_path, index=False)
