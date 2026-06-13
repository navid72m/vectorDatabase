"""Quantization benchmark: memory footprint at matched recall.

Quantized indexes don't compete with HNSW on speed — they compete on *bytes
per vector*. This measures the axis where vecdb's TurboQuant / hybrid index
is supposed to win: how much memory does each index need, and what recall
does it deliver?

Compared:
  - vecdb TQIndex (4-bit, 8-bit) — brute-force over TurboQuant codes
  - vecdb HybridIndex (4-bit, 8-bit) — HNSW over codes + fp32 rerank
  - faiss IndexFlatL2 — fp32 baseline (full memory, exact)
  - faiss IndexScalarQuantizer (8-bit) — faiss's brute-force SQ
  - faiss IndexHNSWPQ — faiss's quantized graph (the hybrid's true rival)

Reports bytes/vector and recall@10. Speed is secondary here and noted only
for context (this script may run on few cores).

Usage:
    PYTHONPATH=. python3 benchmarks/bench_quant_memory.py --n 50000 --dim 128
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from pyvecdb import TQIndex, HybridIndex, VecDB


def recall_at_k(got, gt_sets, k):
    return np.mean([len(set(got[i].tolist()) & gt_sets[i]) / k for i in range(len(got))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--clusters", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    C = rng.uniform(-1, 1, (a.clusters, a.dim)).astype(np.float32)
    data = (C[rng.integers(0, a.clusters, a.n)]
            + 0.15 * rng.standard_normal((a.n, a.dim))).astype(np.float32)
    queries = (C[rng.integers(0, a.clusters, a.queries)]
               + 0.15 * rng.standard_normal((a.queries, a.dim))).astype(np.float32)

    # exact ground truth (chunked to bound memory)
    gt = np.empty((a.queries, a.k), dtype=np.int64)
    for i in range(a.queries):
        d = ((data - queries[i]) ** 2).sum(1)
        gt[i] = np.argpartition(d, a.k)[:a.k][np.argsort(d[np.argpartition(d, a.k)[:a.k]])]
    gt_sets = [set(row.tolist()) for row in gt]

    fp32_bytes = a.dim * 4
    print(f"n={a.n:,} dim={a.dim} k={a.k}  (fp32 = {fp32_bytes} bytes/vector)\n")
    rows = []

    # ---- vecdb TQIndex ----
    for bits in (8, 4):
        tq = TQIndex(a.dim, bits=bits)
        tq.add(data)
        ids, _ = tq.search(queries, k=a.k)
        rec = recall_at_k(ids, gt_sets, a.k)
        bpv = tq.memory_bytes() / a.n
        rows.append((f"vecdb TQIndex {bits}-bit", bpv, rec, fp32_bytes / bpv))

    # ---- vecdb HybridIndex (resident = codes+graph; fp32 mmap-able) ----
    for bits in (8, 4):
        h = HybridIndex(a.dim, bits=bits, ef_construction=200, rerank_mult=4)
        h.add(data)
        ids, _ = h.search(queries, k=a.k, ef=200)
        rec = recall_at_k(ids, gt_sets, a.k)
        bpv = h.memory_bytes(include_fp32=False) / a.n     # resident footprint
        rows.append((f"vecdb Hybrid {bits}-bit (resident)", bpv, rec, fp32_bytes / bpv))

    # ---- faiss baselines ----
    try:
        import faiss
        # fp32 flat (exact, full memory)
        idx = faiss.IndexFlatL2(a.dim); idx.add(data)
        _, fids = idx.search(queries, a.k)
        rows.append(("faiss IndexFlatL2 (fp32)", float(fp32_bytes),
                     recall_at_k(fids, gt_sets, a.k), 1.0))

        # 8-bit scalar quantizer (brute force) — TQIndex's true rival
        sq = faiss.IndexScalarQuantizer(a.dim, faiss.ScalarQuantizer.QT_8bit)
        sq.train(data); sq.add(data)
        _, sids = sq.search(queries, a.k)
        rows.append(("faiss SQ 8-bit", float(a.dim),  # 1 byte/dim
                     recall_at_k(sids, gt_sets, a.k), fp32_bytes / a.dim))

        # HNSWPQ — faiss's quantized graph, the hybrid's true rival
        m_pq = a.dim // 2                       # subquantizers (2 dims each), 8 bits
        try:
            hpq = faiss.IndexHNSWPQ(a.dim, m_pq, 16)
            hpq.hnsw.efConstruction = 200
            hpq.train(data); hpq.add(data)
            hpq.hnsw.efSearch = 200
            _, hids = hpq.search(queries, a.k)
            pq_bpv = m_pq                        # 1 byte per subquantizer code
            rows.append(("faiss IndexHNSWPQ", float(pq_bpv),
                         recall_at_k(hids, gt_sets, a.k), fp32_bytes / pq_bpv))
        except Exception as e:
            print(f"(IndexHNSWPQ skipped: {e})")
    except ImportError:
        print("(faiss not installed — showing vecdb rows only)")

    # ---- table ----
    print(f"{'index':<34} {'bytes/vec':>10} {'recall@'+str(a.k):>10} {'vs fp32':>9}")
    print("-" * 66)
    for name, bpv, rec, ratio in rows:
        print(f"{name:<34} {bpv:>10.1f} {rec:>10.3f} {ratio:>8.1f}x")


if __name__ == "__main__":
    main()
