# list_solvers.py
from __future__ import annotations

import csv
from pathlib import Path

from virne import solver  # noqa: F401
from virne.solver import SolverRegistry

registry = SolverRegistry.list_registered()
rows = [
    (name, getattr(cls, 'type', 'unknown'))
    for name, cls in sorted(registry.items(), key=lambda x: x[0])
]

root_dir = Path(__file__).resolve().parent
txt_path = root_dir / 'solver_list.txt'
csv_path = root_dir / 'solver_list.csv'

txt_path.write_text('\n'.join(name for name, _ in rows) + '\n', encoding='utf-8')

with csv_path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['solver_name', 'solver_type'])
    writer.writerows(rows)

print(f'Wrote {len(rows)} solvers to {txt_path} and {csv_path}')
