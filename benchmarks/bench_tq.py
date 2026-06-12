"""Quantization shoot-out: vecdb TurboQuant vs FAISS quantizers, matched byte budgets.

All methods: brute-force scan over compressed codes (no graph), so the
comparison isolates quantization quality. Re-rank = fetch top-100 by
compressed distance, recompute exact fp32 distance, keep top-10.

Byte budgets at dim=128:
  ~64 B/vec : TQ 4-bit  vs  FAISS SQ4  vs  FAISS PQ64x8 (trained!)
  ~128 B/vec: TQ 8-bit  vs  FAISS SQ8
  512 B/vec : fp32 exact baseline
"""
import time
import numpy as np
import faiss
from pyvecdb import TQIndex

N, DIM, Q, K, RERANK = 20_000, 128, 200, 10, 100
rng = np.random.default_rng(42)

NCLUST = 64
centers = rng.uniform(-1, 1, (NCLUST, DIM)).astype(np.float32)
data = (centers[rng.integers(0, NCLUST, N)] + 0.15 * rng.standard_normal((N, DIM))).astype(np.float32)
queries = (centers[rng.integers(0, NCLUST, Q)] + 0.15 * rng.standard_normal((Q, DIM))).astype(np.float32)

faiss.omp_set_num_threads(1)
flat = faiss.IndexFlatL2(DIM); flat.add(data)
_, gt = flat.search(queries, K)

def recall(found, true=gt):
    return sum(len(set(f.tolist()) & set(t.tolist())) for f, t in zip(found, true)) / (len(true) * K)

def rerank(cand_ids):
    out = np.zeros((Q, K), dtype=np.int64)
    for i in range(Q):
        c = cand_ids[i].astype(np.int64)
        d = ((data[c] - queries[i]) ** 2).sum(1)
        out[i] = c[np.argsort(d)[:K]]
    return out

rows = []

# ---- vecdb TurboQuant -----------------------------------------------------
for bits in (4, 8):
    tq = TQIndex(DIM, bits=bits)
    t0 = time.perf_counter(); tq.add(data); t_build = time.perf_counter() - t0
    bpv = tq.memory_bytes() / N

    t0 = time.perf_counter(); ids, _ = tq.search(queries, k=K); t = time.perf_counter() - t0
    rows.append((f"vecdb TurboQuant {bits}-bit", "no train", bpv, t_build, Q/t,
                 recall(ids.astype(np.int64))))

    t0 = time.perf_counter()
    cids, _ = tq.search(queries, k=RERANK)
    rr = rerank(cids)
    t = time.perf_counter() - t0
    rows.append((f"vecdb TurboQuant {bits}-bit +rerank", "no train", bpv, t_build, Q/t, recall(rr)))

# ---- FAISS scalar quantizers ------------------------------------------------
for name, qt, bpv in (("faiss SQ4", faiss.ScalarQuantizer.QT_4bit, 64),
                      ("faiss SQ8", faiss.ScalarQuantizer.QT_8bit, 128)):
    idx = faiss.IndexScalarQuantizer(DIM, qt, faiss.METRIC_L2)
    t0 = time.perf_counter(); idx.train(data); idx.add(data); t_build = time.perf_counter() - t0
    t0 = time.perf_counter(); _, ids = idx.search(queries, K); t = time.perf_counter() - t0
    rows.append((name, "min/max", bpv, t_build, Q/t, recall(ids)))
    t0 = time.perf_counter(); _, cids = idx.search(queries, RERANK)
    rr = rerank(cids); t = time.perf_counter() - t0
    rows.append((name + " +rerank", "min/max", bpv, t_build, Q/t, recall(rr)))

# ---- FAISS PQ (trained codebooks, same 64 B budget) -------------------------
pq = faiss.IndexPQ(DIM, 64, 8)   # 64 subquantizers x 8 bits = 64 B/vec
t0 = time.perf_counter(); pq.train(data); pq.add(data); t_build = time.perf_counter() - t0
t0 = time.perf_counter(); _, ids = pq.search(queries, K); t = time.perf_counter() - t0
rows.append(("faiss PQ64x8", "TRAINED", 64, t_build, Q/t, recall(ids)))
t0 = time.perf_counter(); _, cids = pq.search(queries, RERANK)
rr = rerank(cids); t = time.perf_counter() - t0
rows.append(("faiss PQ64x8 +rerank", "TRAINED", 64, t_build, Q/t, recall(rr)))

# ---- report -----------------------------------------------------------------
print(f"\nN={N:,} dim={DIM} q={Q} k={K} rerank-depth={RERANK} (single thread, flat scans)\n")
print(f"{'method':<36} {'codebook':<9} {'B/vec':>6} {'build':>7} {'QPS':>8} {'recall@10':>10}")
print("-" * 82)
t0 = time.perf_counter(); flat.search(queries, K); t_fp32 = time.perf_counter() - t0
print(f"{'fp32 exact (faiss flat)':<36} {'-':<9} {512:>6} {'-':>7} {Q/t_fp32:>8,.0f} {1.0:>10.3f}")
for name, cb, bpv, tb, qps, rec in rows:
    print(f"{name:<36} {cb:<9} {bpv:>6.0f} {tb:>6.1f}s {qps:>8,.0f} {rec:>10.3f}")
