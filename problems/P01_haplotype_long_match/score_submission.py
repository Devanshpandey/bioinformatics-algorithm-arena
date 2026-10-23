"""
P01 SEALED SCORER -- the single entry point a search loop is allowed to call.

Contract (SPEC.md Part III):
  * A candidate defines ONLY build() and query_batch()/query().
  * The immutable set -- oracle.py, generate.py, pbwt_core.py, this file -- is
    SHA-256 verified on every run against immutable.lock. A candidate that edits
    the evaluator, the truth source, or the instance generator scores nothing.
  * Gate 1 is exact set equality against the brute-force oracle on FRESHLY
    seeded instances, drawn from BOTH the synthetic and the coalescent
    (scored-distribution) families -- the split that Finding 5 showed is
    necessary, because synthetic panels alone cannot express a multi-allelic or
    distribution-specific bug.
  * Only after gate 1 passes is anything measured.

Objective: INDEX BITS PER PANEL CELL at fixed exact correctness. Space is
scored at the small tier because index size is meaningful at any N. Per
SPEC.md Finding 10 the TIME axis is not comparable below N=10,000, so query
time is reported for information and must never be quoted as a claim here.

Usage:  python3 score_submission.py --module candidates.sub_foo --json
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

IMMUTABLE = ("oracle.py", "generate.py", "pbwt_core.py", "score_submission.py")
LOCK_PATH = os.path.join(HERE, "immutable.lock")


def hashes() -> dict:
    out = {}
    for f in IMMUTABLE:
        with open(os.path.join(HERE, f), "rb") as fh:
            out[f] = hashlib.sha256(fh.read()).hexdigest()
    return out


def check_lock(allow_init: bool = False) -> tuple:
    """Verify the immutable set against immutable.lock.

    HARDENED after a verification agent flagged the original as trust-on-first-
    use: it silently CREATED the lock when missing, so anyone who deleted the
    lock, edited the oracle, and re-ran would have been re-locked to the tampered
    files with a cheerful "ok". Creating the lock now requires an explicit
    --init-lock, and a missing lock is a hard failure.

    Residual weakness, stated rather than hidden: the lock is a local file and
    nothing signs it. Real protection comes from the lock being committed to git
    so any edit to it is visible in history.
    """
    cur = hashes()
    if allow_init:
        # --init-lock (re)writes the lock unconditionally, whether it is missing
        # or merely stale. This is the ONLY path that may write it.
        existed = os.path.exists(LOCK_PATH)
        with open(LOCK_PATH, "w") as fh:
            json.dump(cur, fh, indent=2, sort_keys=True)
        return True, ("lock re-created (explicit --init-lock)" if existed
                      else "lock created (explicit --init-lock)")
    if not os.path.exists(LOCK_PATH):
        return False, ("immutable.lock is MISSING -- refusing to score. "
                       "Re-create it deliberately with --init-lock only if "
                       "you have verified the immutable set against git.")
    ref = json.load(open(LOCK_PATH))
    bad = [f for f in IMMUTABLE if ref.get(f) != cur[f]]
    return (not bad), ("ok" if not bad else "MODIFIED: " + ",".join(bad))


def _run(mod, panel, qh, L):
    idx = mod.build(panel)
    got = set()
    if hasattr(mod, "query_batch"):
        for i, ms in enumerate(mod.query_batch(idx, qh, L)):
            for (j, s, e) in ms:
                got.add((i, j, s, e))
    else:
        for i in range(qh.shape[0]):
            for (j, s, e) in mod.query(idx, qh[i], L):
                got.add((i, j, s, e))
    return got


def gate(mod, base_seed, n_syn=12, n_coal=2):
    from oracle import long_matches
    rng = np.random.default_rng(base_seed)

    for t in range(n_syn):
        N = int(rng.integers(6, 30))
        M = int(rng.integers(10, 50))
        L = int(rng.integers(2, 9))
        f = rng.integers(0, 2, size=(3, M), dtype=np.uint8)
        rows = []
        for _ in range(N + 4):
            h = f[rng.integers(0, 3)].copy()
            for _ in range(int(rng.integers(0, 3))):
                bp = int(rng.integers(0, M))
                h[bp:] = f[rng.integers(0, 3)][bp:]
            rows.append(np.where(rng.random(M) < 0.03, 1 - h, h).astype(np.uint8))
        G = np.array(rows)
        panel, qh = G[:N], G[N:]
        if _run(mod, panel, qh, L) != long_matches(panel, qh, L):
            return False, "synthetic instance %d (N=%d,M=%d,L=%d)" % (t, N, M, L)

    from generate import generate
    for t in range(n_coal):
        panel, qh, _ = generate("tiny", base_seed + 5000 + t)
        if _run(mod, panel, qh[:2], 40) != long_matches(panel, qh[:2], 40):
            return False, "coalescent instance %d" % t
    return True, "exact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--tier", default="small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=int, default=300)
    ap.add_argument("--gate-seed", type=int, default=0,
                    help="0 = draw a fresh gate seed from the clock")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--init-lock", action="store_true",
                    help="deliberately (re)create immutable.lock; never use this "
                         "to make a failing immutable check go away")
    a = ap.parse_args()

    rec = {"module": a.module, "tier": a.tier, "seed": a.seed}
    ok, msg = check_lock(allow_init=a.init_lock)
    rec["immutable_ok"], rec["immutable_msg"] = ok, msg
    if not ok:
        rec["gate"] = False
        rec["error"] = "immutable set violated: " + msg
        print(json.dumps(rec))
        sys.exit(2)

    gseed = a.gate_seed or (int(time.time() * 1000) % 900000)
    rec["gate_seed"] = gseed
    try:
        mod = importlib.import_module(a.module)
        passed, why = gate(mod, gseed)
        rec["gate"], rec["gate_detail"] = passed, why
        if not passed:
            print(json.dumps(rec))
            sys.exit(1)

        from generate import generate
        panel, qh, _ = generate(a.tier, a.seed)
        N, M = panel.shape

        t0 = time.perf_counter()
        idx = mod.build(panel)
        rec["build_s"] = time.perf_counter() - t0

        nb = idx.nbytes() if hasattr(idx, "nbytes") else -1
        rec["index_bytes"] = int(nb)
        rec["bits_per_cell"] = float(nb * 8.0 / (N * M)) if nb >= 0 else None

        t0 = time.perf_counter()
        if hasattr(mod, "query_batch"):
            nm = sum(len(m) for m in mod.query_batch(idx, qh, a.L))
        else:
            nm = sum(len(mod.query(idx, qh[i], a.L)) for i in range(qh.shape[0]))
        rec["query_s_total"] = time.perf_counter() - t0
        rec["query_s_per_query"] = rec["query_s_total"] / qh.shape[0]
        rec["n_matches"] = int(nm)
        rec["N"], rec["M"] = int(N), int(M)
        rec["note_time"] = ("informational only: per SPEC.md Finding 10 the time "
                            "axis is not comparable below N=10,000")
    except Exception:
        rec["gate"] = False
        rec["error"] = traceback.format_exc()[-1500:]
        print(json.dumps(rec))
        sys.exit(1)

    with open(os.path.join(HERE, "search_log.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
