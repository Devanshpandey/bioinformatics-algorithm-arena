"""
P01 SEALED SCORER -- ONLINE (single-query) workload.

score_submission.py scores SPACE on a batch workload, where every submission is
free to sweep the whole panel once and amortise it over many queries. That
workload cannot see the actual open problem, because a sweep is O(N*M) no matter
how well the columns are compressed.

This scorer measures the thing that is open:

    Answer ONE long-match query against a prebuilt index, in time that grows
    slower than O(N*M).

WHY THIS METRIC, AND HOW IT WAS CALIBRATED. Two design choices were forced by
measurement, not taste:

  (1) M IS HELD FIXED across the ladder (--fix-M truncation). M grows with N in
      a coalescent panel, so a fit against N alone silently absorbs the M growth
      and reports a meaningless exponent.

  (2) L IS LARGE (default 2000). Correct output for all-long-match reporting is
      Omega(occ), and occ itself grows as ~N^1.13 at fixed M. At a small L the
      output term therefore FLOORS the exponent and the metric cannot separate
      an O(N*M) sweep from an O(M polylog + occ) index -- both are output-bound.
      At L=2000 the answer set is 7-124 matches across the whole ladder, the
      output term vanishes, and what remains is pure search cost.

Calibrated incumbent (sub_rlpbwt_streaming, M=3000, L=2000, N=2k..16k):
      b = +0.635, R^2 = 0.989, 1.48 s for ONE query at N=15,996.
A structure with O(1) divergence access answers the same query in
O(M polylog + occ) -- with occ ~ constant here, that is time INDEPENDENT of N,
i.e. b ~ 0. So the scale runs from 0.635 (sweep) to ~0 (real index), and a
scaling exponent cannot be faked by tuning a constant factor.

SCORED:
  gate            exact set equality vs the brute-force oracle (unchanged)
  online_exponent fitted b in log(single-query seconds) = a + b*log(N)   <- PRIMARY
  online_s_at_N   absolute single-query seconds at the largest ladder rung
  bits_per_cell   index space, so a submission cannot buy time with unbounded space
  build_s         reported, not scored (an index is built once)

Incumbent b = +0.635. Any submission with a materially lower exponent, a
passing gate, and bounded space is a genuine advance on the open problem.

Usage:
  python3 score_online.py --module candidates.sub_foo --json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import msprime  # noqa: E402

from oracle import long_matches  # noqa: E402

IMMUTABLE = ("oracle.py", "generate.py", "pbwt_core.py", "score_online.py")
LOCK_PATH = os.path.join(HERE, "immutable_online.lock")

RHO, MU, NE = 1e-8, 1.5e-8, 10_000


def hashes() -> dict:
    out = {}
    for f in IMMUTABLE:
        with open(os.path.join(HERE, f), "rb") as fh:
            out[f] = hashlib.sha256(fh.read()).hexdigest()
    return out


def check_lock(allow_init: bool = False) -> tuple:
    cur = hashes()
    if allow_init:
        existed = os.path.exists(LOCK_PATH)
        with open(LOCK_PATH, "w") as fh:
            json.dump(cur, fh, indent=2, sort_keys=True)
        return True, ("lock re-created" if existed else "lock created")
    if not os.path.exists(LOCK_PATH):
        return False, ("immutable_online.lock is MISSING -- refusing to score. "
                       "Re-create deliberately with --init-lock.")
    ref = json.load(open(LOCK_PATH))
    bad = [f for f in IMMUTABLE if ref.get(f) != cur[f]]
    return (not bad), ("ok" if not bad else "MODIFIED: " + ",".join(bad))


def make_panel(n_hap, seq_len, seed, fix_M=None):
    """Panel at a given N, TRUNCATED to a fixed number of sites.

    Holding M fixed is not cosmetic. Calibration showed that letting M grow with
    N (it does: more haplotypes means more segregating sites) confounds the fit,
    because the cost is O(N*M) but the regression is against N alone. Truncating
    every rung to the same M isolates the N dependence, which is the whole point
    of the measurement.
    """
    ts = msprime.sim_ancestry(samples=n_hap // 2, ploidy=2,
                              sequence_length=seq_len, recombination_rate=RHO,
                              population_size=NE, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1,
                               model=msprime.BinaryMutationModel())
    G = ts.genotype_matrix().T.astype(np.uint8)
    if fix_M is not None:
        if G.shape[1] < fix_M:
            raise RuntimeError(f"only {G.shape[1]} sites at N={n_hap}, need {fix_M}")
        G = G[:, :fix_M]
    return G[4:], G[:4]


def _one_query(mod, index, z, L):
    """Exactly one query. A submission with only query_batch is called with a
    batch of size 1 -- which is the honest cost of one online query for it."""
    if hasattr(mod, "query"):
        return mod.query(index, z, L)
    return mod.query_batch(index, z.reshape(1, -1), L)[0]


def gate(mod, base_seed, n_syn=10, n_coal=2, verbose=False):
    """Unchanged from the batch scorer: exactness on fresh synthetic AND
    coalescent instances. Correctness is never a tradeable axis."""
    rng = np.random.default_rng(base_seed)
    for t in range(n_syn):
        N = int(rng.integers(5, 26)); M = int(rng.integers(10, 45))
        L = int(rng.integers(2, 9))
        f = rng.integers(0, 2, size=(3, M), dtype=np.uint8)
        rows = []
        for _ in range(N + 3):
            h = f[rng.integers(0, 3)].copy()
            for _ in range(int(rng.integers(0, 3))):
                bp = int(rng.integers(0, M))
                h[bp:] = f[rng.integers(0, 3)][bp:]
            rows.append(np.where(rng.random(M) < 0.03, 1 - h, h).astype(np.uint8))
        G = np.array(rows)
        panel, qh = G[:N], G[N:]
        want = long_matches(panel, qh, L)
        idx = mod.build(panel)
        got = set()
        for i in range(qh.shape[0]):
            for (j, s, e) in _one_query(mod, idx, qh[i], L):
                got.add((i, j, s, e))
        if got != want:
            return False, f"synthetic instance {t} (N={N} M={M} L={L})"
    for t in range(n_coal):
        panel, qh = make_panel(60, 3e5, base_seed + 7000 + t)
        L = 40
        want = long_matches(panel, qh[:2], L)
        idx = mod.build(panel)
        got = set()
        for i in range(2):
            for (j, s, e) in _one_query(mod, idx, qh[i], L):
                got.add((i, j, s, e))
        if got != want:
            return False, f"coalescent instance {t}"
    return True, "exact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--sizes", default="2000,4000,8000,16000")
    ap.add_argument("--seq-len", type=float, default=1e6)
    ap.add_argument("--fix-M", type=int, default=3000,
                    help="truncate every ladder rung to this many sites, so the "
                         "fit measures scaling in N and not in M")
    ap.add_argument("--L", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--init-lock", action="store_true")
    a = ap.parse_args()

    rec = {"module": a.module, "workload": "online"}
    ok, msg = check_lock(allow_init=a.init_lock)
    rec["immutable_ok"], rec["immutable_msg"] = ok, msg
    if not ok:
        rec["gate"] = False
        rec["error"] = "immutable set violated: " + msg
        print(json.dumps(rec) if a.json else rec)
        sys.exit(1)

    gs = a.gate_seed or int(time.time() * 1000) % 1_000_000
    rec["gate_seed"] = gs
    try:
        mod = importlib.import_module(a.module)
    except Exception as e:
        rec.update(gate=False, error=f"import failed: {e}")
        print(json.dumps(rec) if a.json else rec)
        sys.exit(1)

    try:
        passed, detail = gate(mod, gs)
    except Exception:
        rec.update(gate=False, error="gate raised:\n" + traceback.format_exc()[-600:])
        print(json.dumps(rec) if a.json else rec)
        sys.exit(1)
    rec["gate"], rec["gate_detail"] = passed, detail
    if not passed:
        print(json.dumps(rec) if a.json else rec)
        sys.exit(1)

    ladder = []
    for n in [int(x) for x in a.sizes.split(",")]:
        panel, qh = make_panel(n, a.seq_len, a.seed, fix_M=a.fix_M)
        N, M = panel.shape
        t0 = time.perf_counter()
        idx = mod.build(panel)
        build_s = time.perf_counter() - t0
        ib = idx.nbytes() if hasattr(idx, "nbytes") else -1

        # ONE query, timed. Minimum over repeats (SPEC II.3).
        best, nm = float("inf"), None
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            res = _one_query(mod, idx, qh[0], a.L)
            dt = time.perf_counter() - t0
            nm = len(res)
            best = min(best, dt)
        ladder.append(dict(N=int(N), M=int(M), build_s=build_s,
                           index_bytes=int(ib),
                           bits_per_cell=ib * 8.0 / (N * M) if ib >= 0 else -1,
                           online_s=best, n_matches=int(nm)))

    rec["ladder"] = ladder
    rec["fix_M"] = a.fix_M
    x = np.log(np.array([r["N"] for r in ladder], dtype=float))
    y = np.log(np.array([r["online_s"] for r in ladder], dtype=float))
    b, a0 = np.polyfit(x, y, 1)
    resid = y - (a0 + b * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    rec["online_exponent"] = float(b)
    rec["fit_r2"] = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    rec["online_s_at_max_N"] = ladder[-1]["online_s"]
    rec["bits_per_cell"] = ladder[-1]["bits_per_cell"]
    rec["max_N"] = ladder[-1]["N"]
    # output also grows with N; report the exponent net of it for context
    ym = np.log(np.array([max(r["n_matches"], 1) for r in ladder], dtype=float))
    bm, _ = np.polyfit(x, ym, 1)
    rec["output_exponent"] = float(bm)
    rec["exponent_net_of_output"] = float(b - bm)

    print(json.dumps(rec) if a.json else json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
