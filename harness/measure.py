"""
Arena measurement primitives.

Design rules this file enforces (see ../SPEC.md, "Harness contract"):
  * Wall time is reported as the MINIMUM over repeats, not the mean: the
    minimum is the best estimate of the algorithm's cost, and the noise from
    scheduling is one-sided.
  * Peak RSS is measured for a CHILD PROCESS, so lazily-deferred allocation
    and interpreter overhead cannot be hidden from the accounting.
  * The machine fingerprint is recorded with every result. A runtime number
    without its fingerprint is not comparable and the harness treats it as
    unreported.
  * Algorithmic counters (see COUNTERS) are recorded alongside wall time so
    that an algorithmic advance is distinguishable from an engineering one --
    a Python submission with a better counter profile is a real result even
    when it loses on wall clock to a decade-tuned C incumbent.
"""
from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
import time


def peak_rss_bytes(who=resource.RUSAGE_SELF) -> int:
    """Peak resident set size in BYTES (ru_maxrss units differ by platform)."""
    v = resource.getrusage(who).ru_maxrss
    return int(v) if sys.platform == "darwin" else int(v) * 1024


def machine_fingerprint() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def timed(fn, *args, repeats: int = 3, **kwargs):
    """Run fn repeats times; return (result, min_seconds, all_seconds)."""
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return out, min(times), times


def run_child(argv, timeout=None) -> dict:
    """Run a submission in a child process; return timing + true peak RSS."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return {
        "returncode": proc.returncode,
        "wall_s": wall,
        "peak_rss_bytes": int(max(after - before, after)) * scale,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


class Counters:
    """Implementation-independent work counters.

    These are the arena's defense against the implementation-vs-algorithm
    confound. A submission must increment the counters that apply to its
    problem; the harness reports them next to wall time, and the ALGORITHMIC
    leaderboard ranks on counters while the ENGINEERING leaderboard ranks on
    seconds.
    """

    NAMES = (
        "cells",        # DP cells / states evaluated
        "compares",     # symbol or key comparisons
        "probes",       # index/hash lookups
        "io_bytes",     # bytes read or written
        "index_bytes",  # bytes of persistent index
    )

    def __init__(self):
        self._c = {n: 0 for n in self.NAMES}

    def add(self, name: str, n: int = 1):
        self._c[name] += n

    def as_dict(self) -> dict:
        return dict(self._c)


def emit(record: dict, path: str) -> None:
    """Append one result record to the provenance log (JSONL, append-only)."""
    record = dict(record)
    record["machine"] = machine_fingerprint()
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Instance provenance
# --------------------------------------------------------------------------
def generator_versions() -> dict:
    """Versions of every library that can change what an instance IS."""
    out = {}
    for name in ("msprime", "tskit", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "absent"
    return out


def instance_fingerprint(*arrays, **extra) -> dict:
    """Content hash + generator versions for the scored instance.

    This exists because instance generation is NOT version-stable. Measured
    2026-09-01: msprime 1.3.1/tskit 0.5.7 (laptop) and msprime 1.4.3/tskit 1.0.3
    (HPC) produce DIFFERENT panels from the SAME seed -- P02's tiny tier gives
    7,382 sites on one and 7,384 on the other. A result row without this
    fingerprint cannot be compared across machines, and comparing two such rows
    silently measures the instance difference instead of the algorithm.

    Every evaluator emits this. The scoring step must refuse to compare rows
    whose `sha` differs.
    """
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return {"sha": h.hexdigest()[:16], "versions": generator_versions(), **extra}
