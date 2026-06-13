"""Parallel search scaling benchmark.

Search is read-only over a frozen graph (thread-safe via per-thread visited
buffers), so unlike insert it has no lock contention and should scale close
to linearly with cores. Sweeps thread counts for HNSW, exact, and
TurboQuant batch search and reports QPS + speedup.

Requires an OpenMP build:  make OMP=1 CC=gcc-15
Run:  PYTHONPATH=. python3 benchmarks/bench_search_mt.py --n 500000
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pyvecdb import VecDB, TQIndex, set_threads

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500_000)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--queries", type=int, default=10_000)
p.add_argument("--k", type=int, default=10)
p.add_argument("--ef", type=int, default=100)
p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6, 8])
p.add_argument("--clusters", type=int, default=1024)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
C = rng.uniform(-1, 1, (a.clusters, a.dim)).astype(np.float32)
def gen(n):
    return (C[rng.integers(0, a.clusters, n)]
            + 0.15 * rng.standard_normal((n, a.dim))).astype(np.float32)

print(f"n={a.n:,} dim={a.dim} queries={a.queries:,} k={a.k} ef={a.ef}")
data = gen(a.n)
queries = gen(a.queries)
ids = np.arange(a.n, dtype=np.uint64)

print("building index (threads=8 if available)...")
db = VecDB(dim=a.dim, M=16, ef_construction=200, capacity=a.n)
db.add_bulk(ids, data, threads=8)

def sweep(label, fn):
    print(f"\n{label}: {'threads':>8} {'QPS':>12} {'speedup':>9}")
    base = None
    for t in a.threads:
        set_threads(t)
        best = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            best = max(best, a.queries / (time.perf_counter() - t0))
        if base is None:
            base = best
        print(f"{'':>8} {t:>8} {best:>12,.0f} {best/base:>8.2f}x")

sweep("HNSW", lambda: db.search(queries, k=a.k, ef=a.ef))
sweep("exact (flat)", lambda: db.search(queries[:1000], k=a.k, exact=True))

tq = TQIndex(a.dim, bits=8)
tq.add(data[:min(a.n, 100_000)])             # TQ is brute force; cap for time
sweep("TurboQuant 8-bit", lambda: tq.search(queries[:2000], k=a.k))
