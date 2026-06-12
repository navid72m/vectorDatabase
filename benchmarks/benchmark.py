"""Four-way ANN benchmark: vecdb (ours) vs FAISS vs Chroma vs Qdrant.

Identical data, identical k, ground truth = exact L2 search.
HNSW parameters matched where the system exposes them: M=16, ef_construction=200.

Honesty notes (also printed in output):
- Chroma runs in-process (its normal embedded mode), HNSW under the hood.
- Qdrant runs via qdrant-client local mode (in-process Python/numpy) because no
  server is available here. That mode does EXACT search — representative of
  qdrant-client local prototyping, NOT of a production Qdrant server.
- FAISS HNSW is the closest apples-to-apples comparison to vecdb.
"""
import time
import numpy as np

N, DIM, Q, K = 20_000, 128, 200, 10
M, EF_C = 16, 200
EF_SEARCH = [10, 50, 100, 200]

rng = np.random.default_rng(42)

# Gaussian-mixture data: 64 clusters, sigma 0.15 (mimics embedding manifolds)
NCLUST = 64
centers = rng.uniform(-1, 1, (NCLUST, DIM)).astype(np.float32)
assign = rng.integers(0, NCLUST, N)
data = (centers[assign] + 0.15 * rng.standard_normal((N, DIM))).astype(np.float32)
q_assign = rng.integers(0, NCLUST, Q)
queries = (centers[q_assign] + 0.15 * rng.standard_normal((Q, DIM))).astype(np.float32)

results = []  # (system, config, build_s, qps, recall)


def recall_at_k(found_ids: np.ndarray, true_ids: np.ndarray) -> float:
    hits = sum(len(set(f.tolist()) & set(t.tolist())) for f, t in zip(found_ids, true_ids))
    return hits / (len(true_ids) * K)


# ---------------------------------------------------------------- ground truth
import faiss

flat = faiss.IndexFlatL2(DIM)
flat.add(data)
t0 = time.perf_counter()
_, gt_ids = flat.search(queries, K)
t_flat = time.perf_counter() - t0
results.append(("faiss IndexFlatL2 (exact)", "-", 0.0, Q / t_flat, 1.000))

# ---------------------------------------------------------------- vecdb (ours)
from pyvecdb import VecDB

db = VecDB(dim=DIM, M=M, ef_construction=EF_C)
t0 = time.perf_counter()
db.add(data)
t_build = time.perf_counter() - t0

t0 = time.perf_counter()
e_ids, _ = db.search(queries, k=K, exact=True)
t_e = time.perf_counter() - t0
results.append(("vecdb flat (exact)", "-", t_build, Q / t_e,
                recall_at_k(e_ids.astype(np.int64), gt_ids)))

for ef in EF_SEARCH:
    t0 = time.perf_counter()
    ids, _ = db.search(queries, k=K, ef=ef)
    t = time.perf_counter() - t0
    results.append(("vecdb HNSW", f"ef={ef}", t_build, Q / t,
                    recall_at_k(ids.astype(np.int64), gt_ids)))

# ---------------------------------------------------------------- FAISS HNSW
hnsw = faiss.IndexHNSWFlat(DIM, M)
hnsw.hnsw.efConstruction = EF_C
faiss.omp_set_num_threads(1)  # single-thread everywhere for fairness
t0 = time.perf_counter()
hnsw.add(data)
t_build = time.perf_counter() - t0
for ef in EF_SEARCH:
    hnsw.hnsw.efSearch = ef
    t0 = time.perf_counter()
    _, ids = hnsw.search(queries, K)
    t = time.perf_counter() - t0
    results.append(("faiss HNSW", f"ef={ef}", t_build, Q / t,
                    recall_at_k(ids, gt_ids)))

# ---------------------------------------------------------------- Chroma
import chromadb

client = chromadb.EphemeralClient()
col = client.create_collection(
    "bench",
    configuration={"hnsw": {"space": "l2", "max_neighbors": M,
                            "ef_construction": EF_C, "ef_search": 100}},
)
t0 = time.perf_counter()
B = 5000
for s in range(0, N, B):
    col.add(ids=[str(i) for i in range(s, min(s + B, N))],
            embeddings=data[s:s + B])
t_build = time.perf_counter() - t0

t0 = time.perf_counter()
res = col.query(query_embeddings=queries, n_results=K, include=[])
t = time.perf_counter() - t0
ch_ids = np.array([[int(x) for x in row] for row in res["ids"]])
results.append(("chroma HNSW (in-process)", "ef=100", t_build, Q / t,
                recall_at_k(ch_ids, gt_ids)))

# ---------------------------------------------------------------- Qdrant local
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

qc = QdrantClient(":memory:")
qc.create_collection("bench", vectors_config=VectorParams(size=DIM, distance=Distance.EUCLID))
t0 = time.perf_counter()
for s in range(0, N, B):
    qc.upsert("bench", points=[
        PointStruct(id=i, vector=data[i].tolist())
        for i in range(s, min(s + B, N))
    ])
t_build = time.perf_counter() - t0

t0 = time.perf_counter()
qd_ids = []
for q in queries:
    hits = qc.query_points("bench", query=q.tolist(), limit=K).points
    qd_ids.append([h.id for h in hits])
t = time.perf_counter() - t0
results.append(("qdrant local (in-process, exact)", "-", t_build, Q / t,
                recall_at_k(np.array(qd_ids), gt_ids)))

# ---------------------------------------------------------------- report
print(f"\nN={N:,}  dim={DIM}  queries={Q}  k={K}  (single thread)\n")
print(f"{'system':<34} {'config':<8} {'build':>8} {'QPS':>10} {'recall@10':>10}")
print("-" * 74)
for name, cfg, b, qps, rec in results:
    bs = f"{b:.1f}s" if b else "-"
    print(f"{name:<34} {cfg:<8} {bs:>8} {qps:>10,.0f} {rec:>10.3f}")

print("""
notes:
- faiss HNSW pinned to 1 thread; vecdb is single-threaded by construction.
- chroma = embedded mode (hnswlib inside, plus Python/storage layers).
- qdrant = client local mode, exact numpy search; NOT a Qdrant server.
- vecdb QPS includes Python ctypes overhead per query.""")
