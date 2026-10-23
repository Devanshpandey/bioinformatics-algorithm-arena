"""
P01 evaluation harness.

Two gates, in order. A submission that fails gate 1 is not scored at all --
there is no partial credit for a fast wrong answer.

  GATE 1 (correctness, binary): on `--gate-instances` freshly generated tiny
  instances, the submission's match set must be EXACTLY EQUAL to the oracle's
  (set equality on (query, panel, start, end) tuples). Fresh seeds are drawn
  per evaluation run, so a submission cannot be tuned to a fixed instance.

  GATE 2 (performance, multi-objective): build time, query time, peak RSS,
  and index bytes, reported per tier. Scoring rule in ../../SPEC.md.

Usage:
    python3 evaluate.py --tier small --seed 42 --submission baseline_pbwt
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "harness"))
sys.path.insert(0, os.path.dirname(__file__))

from measure import emit, instance_fingerprint, peak_rss_bytes, timed  # noqa: E402
from oracle import long_matches  # noqa: E402


def correctness_gate(mod, n_instances: int, base_seed: int, verbose=True) -> bool:
    rng = np.random.default_rng(base_seed)
    for t in range(n_instances):
        N = int(rng.integers(6, 30)); M = int(rng.integers(10, 50))
        L = int(rng.integers(2, 9))
        founders = rng.integers(0, 2, size=(3, M), dtype=np.uint8)
        rows = []
        for _ in range(N + 4):
            h = founders[rng.integers(0, 3)].copy()
            for _ in range(int(rng.integers(0, 3))):
                bp = int(rng.integers(0, M))
                h[bp:] = founders[rng.integers(0, 3)][bp:]
            h = np.where(rng.random(M) < 0.03, 1 - h, h).astype(np.uint8)
            rows.append(h)
        G = np.array(rows)
        panel, qh = G[:N], G[N:]
        expected = long_matches(panel, qh, L)
        idx = mod.build(panel)
        got = set()
        for i in range(qh.shape[0]):
            for (j, s, e) in mod.query(idx, qh[i], L):
                got.add((i, j, s, e))
        if got != expected:
            if verbose:
                print(f"GATE 1 FAILED on synthetic instance {t} (N={N} M={M} L={L})")
                print(f"  missing {len(expected - got)}, spurious {len(got - expected)}")
            return False

    # GATE 1b -- instances drawn from the SCORED distribution.
    #
    # This half exists because the synthetic half alone was not enough. The
    # synthetic panels are binary by construction, so they could never expose a
    # multi-allelic bug; meanwhile the msprime-generated scored tier contained
    # allele values > 1 at 0.18% of sites, and the PBWT baseline silently
    # returned 5 spurious and 1 missing match out of 3245. A gate whose instance
    # distribution does not cover the scored distribution certifies nothing.
    from generate import generate as _gen
    for t in range(max(n_instances // 8, 2)):
        panel, qh, _ = _gen("tiny", base_seed + 5000 + t)
        L = 40
        expected = long_matches(panel, qh[:2], L)
        idx = mod.build(panel)
        got = set()
        for i in range(2):
            for (j, s, e) in mod.query(idx, qh[i], L):
                got.add((i, j, s, e))
        if got != expected:
            if verbose:
                print(f"GATE 1b FAILED on coalescent instance {t}: "
                      f"missing {len(expected-got)}, spurious {len(got-expected)}")
            return False

    if verbose:
        print(f"GATE 1 PASSED: exact on {n_instances} synthetic + "
              f"{max(n_instances//8,2)} coalescent instances")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=int, default=200, help="min match length in sites")
    ap.add_argument("--submission", default="baseline_pbwt")
    ap.add_argument("--gate-instances", type=int, default=25)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--log", default="results.jsonl")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="skip the full-instance oracle cross-check (large tiers)")
    a = ap.parse_args()

    mod = importlib.import_module(a.submission)

    if not correctness_gate(mod, a.gate_instances, a.seed + 9973):
        emit({"problem": "P01", "submission": a.submission, "gate1": False}, a.log)
        sys.exit(1)

    from generate import generate
    panel, qh, meta = generate(a.tier, a.seed)
    print(f"instance: panel={panel.shape} query={qh.shape} L={a.L} seed={a.seed}")

    idx, build_s, _ = timed(mod.build, panel, repeats=1)
    rss_after_build = peak_rss_bytes()

    batched = hasattr(mod, "query_batch")

    def run_queries():
        if batched:
            return sum(len(m) for m in mod.query_batch(idx, qh, a.L))
        return sum(len(mod.query(idx, qh[i], a.L)) for i in range(qh.shape[0]))

    n_matches, query_s, all_q = timed(run_queries, repeats=a.repeats)

    index_bytes = idx.nbytes() if hasattr(idx, "nbytes") else -1
    rec = {
        "problem": "P01", "submission": a.submission, "gate1": True,
        "tier": a.tier, "seed": a.seed, "L": a.L,
        "n_panel": int(panel.shape[0]), "n_sites": int(panel.shape[1]),
        "n_query": int(qh.shape[0]), "n_matches": int(n_matches),
        "build_s": build_s, "query_s_total": query_s,
        "query_s_per_query": query_s / max(qh.shape[0], 1),
        "peak_rss_bytes": rss_after_build, "index_bytes": index_bytes,
        "index_bytes_per_panel_cell": index_bytes / (panel.shape[0] * panel.shape[1]),
        "repeats": a.repeats, "query_times": all_q, "batched_api": batched,
        "instance": instance_fingerprint(panel, qh, tier=a.tier, seed=a.seed),
    }
    emit(rec, a.log)
    print(json.dumps({k: v for k, v in rec.items() if k != "query_times"}, indent=2))


if __name__ == "__main__":
    main()
