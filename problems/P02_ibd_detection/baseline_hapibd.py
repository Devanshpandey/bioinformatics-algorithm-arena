"""
P02 incumbent: seed-and-extend IBD detection (the hap-IBD design).

Pipeline stages -- these are the mutable surface an agent may replace:

    STAGE 1  seed     find exact all-pairs maximal matches >= seed_cm
    STAGE 2  filter   drop seeds below the seed length threshold
    STAGE 3  extend   merge seeds across short gaps, tolerating <= max_discord
                      discordant sites (genotype error and recurrent mutation
                      break exact matches; real IBD survives them)
    STAGE 4  report   emit merged segments >= min_cm

Stage 1 is the expensive one and is *exactly* P01's problem. That is the whole
point of pairing these two: an agent that improves the P01 index improves P02's
dominant cost without touching P02's logic, and the arena measures it.

Submission API:
    call_ibd(panel, pos, params) -> list of (i, j, left_bp, right_bp)
"""
from __future__ import annotations

import numpy as np

from panel_matches import pbwt_all_matches

RHO = 1e-8
DEFAULTS = dict(seed_cm=1.0, min_cm=2.0, max_gap_cm=0.5, max_discord=2)


def bp_to_cm(bp):
    return 100.0 * RHO * bp


def cm_to_bp(cm):
    return cm / (100.0 * RHO)


def call_ibd(panel: np.ndarray, pos: np.ndarray, params: dict = None):
    p = dict(DEFAULTS, **(params or {}))
    M = panel.shape[1]

    # STAGE 1+2 -- seeds. Convert the cM seed threshold to a site count using
    # the mean marker spacing, then re-check each seed in true cM.
    span_bp = float(pos[-1] - pos[0]) if M > 1 else 1.0
    sites_per_cm = M / max(bp_to_cm(span_bp), 1e-9)
    seed_sites = max(int(p["seed_cm"] * sites_per_cm), 2)

    raw = pbwt_all_matches(panel, seed_sites)
    by_pair = {}
    for (i, j, s, e) in raw:
        lo, hi = pos[s], pos[e - 1] if e <= M else pos[-1]
        if bp_to_cm(hi - lo) >= p["seed_cm"]:
            by_pair.setdefault((i, j), []).append((lo, hi, s, e))

    # STAGE 3 -- extend / merge across short, low-discordance gaps
    max_gap_bp = cm_to_bp(p["max_gap_cm"])
    out = []
    for (i, j), segs in by_pair.items():
        segs.sort()
        cur_lo, cur_hi, _, cur_e = segs[0]
        for (lo, hi, s, e) in segs[1:]:
            gap_bp = lo - cur_hi
            if gap_bp <= max_gap_bp:
                disc = int(np.count_nonzero(
                    panel[i, cur_e:s] != panel[j, cur_e:s])) if s > cur_e else 0
                if disc <= p["max_discord"]:
                    cur_hi, cur_e = max(cur_hi, hi), max(cur_e, e)
                    continue
            if bp_to_cm(cur_hi - cur_lo) >= p["min_cm"]:
                out.append((int(i), int(j), float(cur_lo), float(cur_hi)))
            cur_lo, cur_hi, cur_e = lo, hi, e
        if bp_to_cm(cur_hi - cur_lo) >= p["min_cm"]:
            out.append((int(i), int(j), float(cur_lo), float(cur_hi)))
    return out
