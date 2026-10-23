"""
P02 evaluation harness.

The metric design resolves the two failure modes the audit flagged:

  FRAGMENTATION. A true 10 cM segment reported as two 4 cM pieces is neither a
  clean hit nor a clean miss. Segment-level power alone therefore misreports it.
  We score cM-level recall and precision as the PRIMARY metrics -- overlap
  measured in genetic distance, per pair -- and report segment-level power at
  several reciprocal-overlap thresholds as secondary. Fragmentation costs a
  little cM-level precision instead of silently zeroing a true positive.

  DEGENERATE OPTIMA. Calling nothing maximises precision; calling everything
  maximises recall. Neither passes: the gate requires BOTH above a floor, and
  the reported score is the pair, never a single number.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "harness"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure import emit, instance_fingerprint, peak_rss_bytes, timed  # noqa: E402

RHO = 1e-8
cm = lambda bp: 100.0 * RHO * bp  # noqa: E731


def _by_pair(rows):
    d = {}
    for (i, j, l, r) in rows:
        d.setdefault((int(i), int(j)), []).append((float(l), float(r)))
    for k in d:
        d[k].sort()
    return d


def _merge(iv):
    out = []
    for l, r in sorted(iv):
        if out and l <= out[-1][1]:
            out[-1][1] = max(out[-1][1], r)
        else:
            out.append([l, r])
    return out


def _overlap(a, b):
    """Total overlap length between two interval lists."""
    tot, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def score(truth, calls):
    T, C = _by_pair(truth), _by_pair(calls)
    Tm = {k: _merge(v) for k, v in T.items()}
    Cm = {k: _merge(v) for k, v in C.items()}

    true_cm = sum(cm(r - l) for v in Tm.values() for l, r in v)
    call_cm = sum(cm(r - l) for v in Cm.values() for l, r in v)
    hit_cm = sum(cm(_overlap(Tm[k], Cm[k])) for k in Tm.keys() & Cm.keys())

    recall = hit_cm / true_cm if true_cm else float("nan")
    precision = hit_cm / call_cm if call_cm else float("nan")

    seg_power = {}
    for thr in (0.5, 0.8):
        hits = 0
        for k, segs in T.items():
            cs = Cm.get(k, [])
            for (l, r) in segs:
                ov = _overlap([[l, r]], cs)
                if r > l and ov / (r - l) >= thr:
                    hits += 1
        seg_power[f"seg_power_ro{thr}"] = hits / len(truth) if len(truth) else float("nan")

    len_err = []
    for k, segs in T.items():
        cs = Cm.get(k, [])
        for (l, r) in segs:
            best = max((min(r, b) - max(l, a) for a, b in cs), default=0.0)
            if best > 0:
                mc = max((cm(b - a) for a, b in cs if min(r, b) > max(l, a)), default=0.0)
                len_err.append(mc - cm(r - l))

    return dict(
        cm_recall=recall, cm_precision=precision,
        true_cm=true_cm, called_cm=call_cm, hit_cm=hit_cm,
        n_true=len(truth), n_calls=len(calls),
        median_length_error_cm=float(np.median(len_err)) if len_err else float("nan"),
        **seg_power)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--submission", default="baseline_hapibd")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--log", default="results.jsonl")
    ap.add_argument("--recall-floor", type=float, default=0.30)
    ap.add_argument("--precision-floor", type=float, default=0.50)
    a = ap.parse_args()

    mod = importlib.import_module(a.submission)
    from generate import generate

    panel, pos, truth, meta = generate(a.tier, a.seed)
    print(f"instance: panel={panel.shape} sites={len(pos)} true_segments={len(truth)}")

    calls, secs, _ = timed(mod.call_ibd, panel, pos, repeats=a.repeats)
    s = score([tuple(t) for t in truth], calls)

    gate = (s["cm_recall"] >= a.recall_floor) and (s["cm_precision"] >= a.precision_floor)
    rec = dict(problem="P02", submission=a.submission, tier=a.tier, seed=a.seed,
               wall_s=secs, peak_rss_bytes=peak_rss_bytes(), gate=bool(gate),
               n_hap=int(panel.shape[0]), n_sites=int(panel.shape[1]),
               instance=instance_fingerprint(panel, truth, tier=a.tier, seed=a.seed), **s)
    emit(rec, a.log)
    print(json.dumps(rec, indent=2, default=float))
    print("\nGATE", "PASSED" if gate else "FAILED",
          f"(recall>={a.recall_floor}, precision>={a.precision_floor})")


if __name__ == "__main__":
    main()
