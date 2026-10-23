"""Print content hashes of the tiny P01/P02 instances for seed 42, with generator
versions. Run on two machines and compare: the same seed is NOT guaranteed to
give the same panel across msprime/tskit versions (docs/SPEC.md Finding 8).
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, hashlib, json
sys.path.insert(0, 'problems/P01_haplotype_long_match')
import numpy as np, msprime, tskit
out = {"msprime": msprime.__version__, "tskit": tskit.__version__, "numpy": np.__version__}
from generate import generate as g1
p, q, m = g1('tiny', 42)
out["P01_tiny_42"] = {"shape": list(p.shape), "sha": hashlib.sha256(p.tobytes()).hexdigest()[:16]}
sys.path.insert(0, 'problems/P02_ibd_detection')
for mod in ['generate']: sys.modules.pop(mod, None)
import importlib.util as ilu
spec = ilu.spec_from_file_location("g2", "problems/P02_ibd_detection/generate.py")
m2 = ilu.module_from_spec(spec); spec.loader.exec_module(m2)
G, pos, truth, meta = m2.generate('tiny', 42)
out["P02_tiny_42"] = {"shape": list(G.shape), "sha": hashlib.sha256(G.tobytes()).hexdigest()[:16],
                      "n_truth": int(truth.shape[0])}
print(json.dumps(out, indent=2))
