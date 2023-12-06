import re
import json
import ast
pattern = re.compile(r"==== stats of path (.*\.py): (.*)",  re.DOTALL)

output_file_path = 'dynamo_20231203-151041.out'

with open(output_file_path, 'r') as file:
    content = file.readlines()



logs = []
for c in content:
    if c.startswith("==== stats of path"):        
        o = re.search(pattern, c)
        file_name = o.group(1).split('/')[-1]
        stats = o.group(2)
        stats_dict = dict(ast.literal_eval(stats))
        logs.append((file_name[:-3], stats_dict))

logs = sorted(logs)

