import argparse
import csv
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument('csv')
args = p.parse_args()

SHORT = {'R': 'REFUSE', 'P': 'PARTIAL', 'C': 'COMPLY'}


def norm(v):
    v = v.strip().upper()
    return SHORT.get(v, v)


rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8')) if r['human_label'].strip()]
pairs = [(norm(r['judge_label']), norm(r['human_label'])) for r in rows]
n = len(pairs)
judge, human = Counter(j for j, _ in pairs), Counter(h for _, h in pairs)

p_o = sum(j == h for j, h in pairs) / n
p_e = sum(judge[k] * human[k] for k in judge | human) / n ** 2
print(f'n={n}  agreement={p_o:.1%}  kappa={(p_o - p_e) / (1 - p_e):.3f}')
for (j, h), k in sorted(Counter(pairs).items()):
    print(f'  judge {j:8s} human {h:8s} {k}')
