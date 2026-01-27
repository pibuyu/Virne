# list_solvers.py
from virne import solver  # noqa: F401
from virne.solver import SolverRegistry

registry = SolverRegistry.list_registered()
for name, cls in sorted(registry.items(), key=lambda x: x[0]):
    print(f"{name}\t{getattr(cls, 'type', 'unknown')}")
