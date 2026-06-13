"""SIFT1M recall-QPS Pareto + multi-threaded head-to-head vs FAISS.

Produces the ann-benchmarks-style figure (recall@10 on x, QPS on y, log
scale) that is the standard way ANN results are communicated, plus a
matched-thread-count comparison so the claim isn't limited to single-thread.

Recall is measured against SIFT1M's provided ground truth.

Download:
    wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz
Run:
    PYTHONPATH=. python3 benchmarks/bench_sift_pareto.py --dir sift --faiss \\
        --search-threads 1 8 --plot sift_pareto.png
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pyvecdb import VecDB, set_threads
from bench_sift import read_fvecs, read_ivecs


def measure(search_fn, queries, gt_sets, k, reps=3):
    """Best-of-reps QPS + recall@k for a search callable."""
    best_qps = 0.0
    res = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = search_fn()
        best_qps = max(best_qps, len(queries) / (time.perf_counter() - t0))
    rec = np.mean([len(set(res[i].tolist()) & gt_sets[i]) / k
                   for i in range(len(queries))])
    return best_qps, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="sift")
    p.add_argument("--prefix", default=None,
                   help="filename prefix (default: basename of --dir, e.g. sift/gist)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--efs", type=int, nargs="+", default=[10, 20, 50, 100, 200, 400, 800])
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--build-threads", type=int, default=8)
    p.add_argument("--search-threads", type=int, nargs="+", default=[1])
    p.add_argument("--faiss", action="store_true")
    p.add_argument("--plot", default="sift_pareto.png")
    args = p.parse_args()

    pfx = args.prefix or os.path.basename(os.path.normpath(args.dir))
    base = read_fvecs(os.path.join(args.dir, f"{pfx}_base.fvecs"))
    query = read_fvecs(os.path.join(args.dir, f"{pfx}_query.fvecs"))
    gt = read_ivecs(os.path.join(args.dir, f"{pfx}_groundtruth.ivecs"))
    n, dim = base.shape
    gt_sets = [set(row[:args.k].tolist()) for row in gt]
    print(f"{pfx.upper()}: {n:,}x{dim}, {len(query):,} queries, k={args.k}")

    ids = np.arange(n, dtype=np.uint64)
    db = VecDB(dim=dim, M=args.M, ef_construction=args.ef_construction, capacity=n)
    t0 = time.perf_counter()
    db.add_bulk(ids, base, threads=args.build_threads)
    print(f"vecdb build: {time.perf_counter()-t0:.1f}s (threads={args.build_threads})")

    # curves[label] = (recalls[], qps[])
    curves = {}

    for T in args.search_threads:
        set_threads(T)
        recs, qpss = [], []
        for ef in args.efs:
            q, r = measure(lambda ef=ef: db.search(query, k=args.k, ef=ef)[0],
                           query, gt_sets, args.k)
            recs.append(r); qpss.append(q)
            print(f"vecdb  T={T:<2} ef={ef:<4} recall={r:.4f} qps={q:,.0f}")
        curves[f"vecdb ({T}t)"] = (recs, qpss)

    if args.faiss:
        import faiss
        idx = faiss.IndexHNSWFlat(dim, args.M)
        idx.hnsw.efConstruction = args.ef_construction
        t0 = time.perf_counter(); idx.add(base)
        print(f"faiss build: {time.perf_counter()-t0:.1f}s")
        for T in args.search_threads:
            faiss.omp_set_num_threads(T)
            recs, qpss = [], []
            for ef in args.efs:
                idx.hnsw.efSearch = ef
                def fsearch(ef=ef):
                    _, fids = idx.search(query, args.k); return fids
                q, r = measure(fsearch, query, gt_sets, args.k)
                recs.append(r); qpss.append(q)
                print(f"faiss  T={T:<2} ef={ef:<4} recall={r:.4f} qps={q:,.0f}")
            curves[f"faiss ({T}t)"] = (recs, qpss)

    # ---- plot: recall@k (x) vs QPS (y, log) — higher and righter is better ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.5))
        styles = {"vecdb": dict(marker="o", lw=2),
                  "faiss": dict(marker="s", lw=2, ls="--")}
        for label, (recs, qpss) in curves.items():
            key = "vecdb" if label.startswith("vecdb") else "faiss"
            order = np.argsort(recs)
            ax.plot(np.array(recs)[order], np.array(qpss)[order],
                    label=label, **styles[key])
        ax.axvspan(0.95, 1.0, color="green", alpha=0.06)
        ax.set_yscale("log")
        ax.set_xlabel(f"recall@{args.k}  (higher-right is better)")
        ax.set_ylabel("QPS (log scale)")
        ax.set_title(f"SIFT1M: recall vs throughput (M={args.M}, efC={args.ef_construction})")
        ax.annotate("typical operating range (recall ≥ 0.95)",
                    xy=(0.975, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8, color="green")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="solid=vecdb  dashed=faiss")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nplot written to {args.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
