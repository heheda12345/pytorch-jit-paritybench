import re
import os
import matplotlib.pyplot as plt


class Summary:
    def __init__(self, log_name, num_graph, num_graph_break, num_ops, break_reasons):
        self.log_name = log_name
        self.num_graph = num_graph
        self.num_graph_break = num_graph_break
        self.num_ops = num_ops
        self.break_reasons = break_reasons


def parse(f_name):
    with open(f_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        if lines[-1] != "finish":
            print(f_name, "not finish")
        # parse it
        matches = re.findall(r"Dynamo produced (\d+) graphs with (-?\d+) graph break and (\d+) ops", lines[0])
        assert len(matches) == 1
        num_graph, num_graph_break, num_ops = matches[0]
        num_graph, num_graph_break, num_ops = int(num_graph), int(num_graph_break), int(num_ops)
        break_detected = 0
        break_reasons = []
        for line in lines:
            if line.startswith("++break_reason: "):
                break_detected += 1
                break_reasons.append(line[len("++break_reason: "):])
        assert break_detected == num_graph
        
        return Summary(f_name, num_graph, num_graph_break, num_ops, break_reasons)


s = []
for f_name in os.listdir("logs"):
    if f_name.endswith(".log"):
        s.append(parse(os.path.join("logs", f_name)))

# sort s by num_graph_break
s.sort(key=lambda x: x.num_graph_break, reverse=True)
print("top10:")
lst = [0 for _ in range(20)]
lst_small = [0 for _ in range(10)]
for i in range(100):
    print("RANK", i)
    print(s[i].log_name, s[i].num_graph_break, s[i].num_ops, s[i].break_reasons)
    print("=================================================")
for ss in s:
    lst[ss.num_graph // 10] += 1
    if ss.num_graph < 10:
        lst_small[ss.num_graph] += 1
print(lst)
print(lst_small)
print("avg graph:", sum([ss.num_graph for ss in s]) / len(s))
