"""SIFT1M benchmark — the standard ANN dataset, with provided ground truth.

SIFT1M (Inria TEXMEX corpus): 1,000,000 base vectors + 10,000 queries, each
128-d, plus a ground-truth file giving the true 100 nearest neighbors per
query. Recall is measured against THAT ground truth (not our own exact
search), so the numbers are directly comparable to published FAISS / hnswlib
/ Qdrant results.

Download (≈170 MB tar, expands to ≈500 MB):
    wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    tar xzf sift.tar.gz          # -> sift/{sift_base,sift_query,sift_groundtruth}.{fvecs,ivecs}

Also works on GIST1M (960-d, same format, ~3.6 GB) and any TEXMEX dataset:
    wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz && tar xzf gist.tar.gz
    python3 benchmarks/bench_sift.py --dir gist          # prefix auto-detected

Run:
    PYTHONPATH=. python3 benchmarks/bench_sift.py --dir sift
    PYTHONPATH=. python3 benchmarks/bench_sift.py --dir sift --faiss --threads 8
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pyvecdb import VecDB, set_threads


def read_fvecs(path):
    """.fvecs: each vector is [int32 dim][dim * float32]. Returns (n, dim) float32."""
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        raise ValueError(f"empty file: {path}")
    dim = a[0]
    a = a.reshape(-1, dim + 1)
    return np.ascontiguousarray(a[:, 1:].view(np.float32))


def read_ivecs(path):
    """.ivecs: same layout but int32 payload (the ground-truth neighbor ids)."""
    a = np.fromfile(path, dtype=np.int32)
    dim = a[0]
    return np.ascontiguousarray(a.reshape(-1, dim + 1)[:, 1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="sift", help="dir with <prefix>_*.{fvecs,ivecs}")
    p.add_argument("--prefix", default=None,
                   help="filename prefix (default: basename of --dir, e.g. sift/gist)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--efs", type=int, nargs="+", default=[10, 50, 100, 200, 400])
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--build-threads", type=int, default=8)
    p.add_argument("--search-threads", type=int, default=1,
                   help="OpenMP threads for the search QPS measurement")
    p.add_argument("--faiss", action="store_true")
    args = p.parse_args()

    pfx = args.prefix or os.path.basename(os.path.normpath(args.dir))
    base = read_fvecs(os.path.join(args.dir, f"{pfx}_base.fvecs"))
    query = read_fvecs(os.path.join(args.dir, f"{pfx}_query.fvecs"))
    gt = read_ivecs(os.path.join(args.dir, f"{pfx}_groundtruth.ivecs"))
    n, dim = base.shape
    nq = query.shape[0]
    print(f"{pfx.upper()}: base={n:,}x{dim}  queries={len(query):,}  k={args.k}  "
          f"M={args.M} efC={args.ef_construction}")
    gt_sets = [set(row[:args.k].tolist()) for row in gt]

    ids = np.arange(n, dtype=np.uint64)
    db = VecDB(dim=dim, M=args.M, ef_construction=args.ef_construction, capacity=n)
    t0 = time.perf_counter()
    db.add_bulk(ids, base, threads=args.build_threads)
    print(f"build: {time.perf_counter()-t0:.1f}s "
          f"(threads={args.build_threads}, {n/(time.perf_counter()-t0):,.0f} vec/s)")

    set_threads(args.search_threads)
    print(f"\n{'index':<14} {'ef':>5} {'QPS':>10} {'recall@'+str(args.k):>10}  "
          f"(search threads={args.search_threads})")
    for ef in args.efs:
        best = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            res, _ = db.search(query, k=args.k, ef=ef)
            best = max(best, nq / (time.perf_counter() - t0))
        rec = np.mean([len(set(res[i].tolist()) & gt_sets[i]) / args.k
                       for i in range(nq)])
        print(f"{'vecdb HNSW':<14} {ef:>5} {best:>10,.0f} {rec:>10.3f}")

    if args.faiss:
        import faiss
        faiss.omp_set_num_threads(args.search_threads)
        idx = faiss.IndexHNSWFlat(dim, args.M)
        idx.hnsw.efConstruction = args.ef_construction
        t0 = time.perf_counter()
        idx.add(base)
        print(f"\nfaiss build: {time.perf_counter()-t0:.1f}s")
        for ef in args.efs:
            idx.hnsw.efSearch = ef
            best = 0.0
            for _ in range(3):
                t0 = time.perf_counter()
                _, fids = idx.search(query, args.k)
                best = max(best, nq / (time.perf_counter() - t0))
            rec = np.mean([len(set(fids[i].tolist()) & gt_sets[i]) / args.k
                           for i in range(nq)])
            print(f"{'faiss HNSW':<14} {ef:>5} {best:>10,.0f} {rec:>10.3f}")


if __name__ == "__main__":
    main()
