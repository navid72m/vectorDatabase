"""Large-scale vecdb benchmark with proper methodology.

- Held-out queries (never database vectors, so recall isn't flattered
  by trivial self-retrieval).
- Choice of data geometry: 'clustered' (Gaussian mixture, mimics real
  embeddings) or 'uniform' (distance-concentration stress test — expect
  much lower recall at scale; that is the data, not the index).
- Optional --faiss A/B: builds faiss HNSW with identical parameters on
  the identical arrays, the definitive implementation-vs-data control.

Usage:
  python benchmarks/bench_large.py --n 1000000 --data clustered
  python benchmarks/bench_large.py --n 1000000 --data uniform --faiss
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pyvecdb import VecDB

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1_000_000)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--data", choices=["clustered", "uniform"], default="clustered")
p.add_argument("--queries", type=int, default=200)
p.add_argument("--k", type=int, default=10)
p.add_argument("--efs", type=int, nargs="+", default=[10, 50, 100, 200, 400])
p.add_argument("--M", type=int, default=16)
p.add_argument("--ef-construction", type=int, default=200)
p.add_argument("--clusters", type=int, default=1024)
p.add_argument("--faiss", action="store_true", help="A/B against faiss HNSW on identical data")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
print(f"data={a.data} n={a.n:,} dim={a.dim} queries={a.queries} (held out) "
      f"M={a.M} efC={a.ef_construction}")

if a.data == "clustered":
    centers = rng.uniform(-1, 1, (a.clusters, a.dim)).astype(np.float32)
    def gen(n):
        return (centers[rng.integers(0, a.clusters, n)]
                + 0.15 * rng.standard_normal((n, a.dim))).astype(np.float32)
else:
    def gen(n):
        return rng.random((n, a.dim), dtype=np.float32)

data = gen(a.n)
queries = gen(a.queries)                       # held out, never inserted
ids = np.arange(a.n, dtype=np.uint64)

db = VecDB(dim=a.dim, M=a.M, ef_construction=a.ef_construction, capacity=a.n)
t0 = time.perf_counter()
db.add_bulk(ids, data)
tb = time.perf_counter() - t0
print(f"insert: {tb:.1f}s  ({a.n/tb:,.0f} vec/s)")

t0 = time.perf_counter()
gt_ids, _ = db.search(queries, k=a.k, exact=True)
tf = time.perf_counter() - t0
gt = [set(int(x) for x in row) for row in gt_ids]
print(f"flat exact (ground truth): {tf:.2f}s  ({a.queries/tf:,.1f} qps)")

print(f"\n{'index':<22} {'ef':>5} {'QPS':>9} {'recall@'+str(a.k):>10}")
for ef in a.efs:
    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        res, _ = db.search(queries, k=a.k, ef=ef)
        best = max(best, a.queries / (time.perf_counter() - t0))
    rec = np.mean([len(set(int(x) for x in res[i]) & gt[i]) / a.k
                   for i in range(a.queries)])
    print(f"{'vecdb HNSW':<22} {ef:>5} {best:>9,.0f} {rec:>10.3f}")

if a.faiss:
    try:
        import faiss
    except ImportError:
        sys.exit("\n--faiss requested but faiss not installed (pip install faiss-cpu)")
    faiss.omp_set_num_threads(1)
    idx = faiss.IndexHNSWFlat(a.dim, a.M)
    idx.hnsw.efConstruction = a.ef_construction
    t0 = time.perf_counter()
    idx.add(data)
    print(f"\nfaiss build: {time.perf_counter()-t0:.1f}s (single thread)")
    for ef in a.efs:
        idx.hnsw.efSearch = ef
        best = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            _, fids = idx.search(queries, a.k)
            best = max(best, a.queries / (time.perf_counter() - t0))
        rec = np.mean([len(set(int(x) for x in fids[i]) & gt[i]) / a.k
                       for i in range(a.queries)])
        print(f"{'faiss HNSW':<22} {ef:>5} {best:>9,.0f} {rec:>10.3f}")
