# vecdb — a vector database from scratch in C

~600 lines of C11, zero dependencies. Exact flat search + HNSW approximate
nearest neighbor (Malkov & Yashunin 2016), binary persistence, deterministic
builds via seeded xorshift RNG.

## Results (50k vectors, dim 128, Gaussian-mixture data, single thread)

| index          | QPS    | recall@10 |
|----------------|--------|-----------|
| flat (exact)   |    218 | 1.000     |
| hnsw ef=10     |  6,480 | 0.759     |
| hnsw ef=50     |  2,895 | 0.990     |
| hnsw ef=100    |  2,427 | 0.999     |
| hnsw ef=200    |  1,953 | 1.000     |

~11x speedup over exact search at 99.9% recall.

## Build & run



## Usage

### Python bindings

```python
from pyvecdb import VecDB

# Create a new DB (dimensionality must match your vectors)

db = VecDB(dim=128, M=16, ef_construction=200)

# Add vectors (numpy array of shape (n, dim), dtype float32)
# You can also add a list of vectors or a (ids, vectors) tuple.
ids = db.add(vectors)  # returns list of assigned IDs

# Search for nearest neighbours
query = np.random.rand(1, 128).astype(np.float32)
ids, distances = db.search(query, k=10, ef=100)
print("Nearest IDs:", ids)
print("Distances:", distances)

# Save / load the index

db.save("index.vecdb")

# Load later
loaded = VecDB.load("index.vecdb")
```

### C API (advanced)

The underlying C library `libvecdb.so` provides the same functionality for
embedding‑heavy workloads. See `vecdb.h` for the full interface.

```c
VecDBConfig cfg = vecdb_default_config(128);
VecDB *db = vecdb_create(&cfg);

// Add a single vector (id must be unique)
float vec[128]; // fill with your data
vecdb_add(db, 42, vec);

// Search
VecResult results[10];
vecdb_search_hnsw(db, query_vec, 10, 100, results);

// Persist
vecdb_save(db, "index.vecdb");
```

## Build & run



## API




## API

```c
VecDBConfig cfg = vecdb_default_config(128);
VecDB *db = vecdb_create(&cfg);
vecdb_add(db, /*id=*/42, vec);
VecResult out[10];
vecdb_search_hnsw(db, query, /*k=*/10, /*ef=*/100, out);
vecdb_save(db, "index.vecdb");
```

## Implementation notes

- **Distance kernel**: 4-way unrolled squared-L2 with `restrict`, written so
  `-O3 -march=native` auto-vectorizes it.
- **HNSW**: geometric level sampling (mL = 1/ln M), greedy descent on upper
  layers, beam search (`ef`) on layer 0, heuristic neighbor selection
  (Algorithm 4) with backfill, bidirectional links pruned to M / 2M caps.
- **Visited set**: epoch-tagged uint32 array — no per-query memset.
- **Heaps**: one array-backed binary heap doing double duty (min for the
  candidate frontier, max for the result set).
- **Persistence**: flat binary format — header, ids, vectors, adjacency lists.
  Load is validated (truncation / corrupt link counts fail cleanly).

## Known limits / roadmap

- No deletes (HNSW deletion needs tombstones + repair)
- Single-threaded build (insert is parallelizable with per-node locks)
- No SIMD intrinsics yet (explicit AVX2/NEON kernel is the obvious next step)
- No quantization (PQ/SQ8 would cut memory 4-32x)

## Comparative benchmark (Python, single thread, N=20k, dim=128, k=10)

Identical Gaussian-mixture data; HNSW params matched (M=16, ef_construction=200);
recall measured against exact L2 ground truth.

| system                          | config | build | QPS    | recall@10 |
|---------------------------------|--------|-------|--------|-----------|
| faiss IndexFlatL2 (exact)       | —      | —     |  2,085 | 1.000     |
| **vecdb HNSW (ours)**           | ef=10  | 7.4s  | 18,347 | 0.897     |
| **vecdb HNSW (ours)**           | ef=50  | 7.4s  | 10,481 | 1.000     |
| **vecdb HNSW (ours)**           | ef=100 | 7.4s  |  8,154 | 1.000     |
| faiss HNSW                      | ef=50  | 3.2s  | 20,138 | 0.999     |
| faiss HNSW                      | ef=100 | 3.2s  | 15,505 | 1.000     |
| chroma HNSW (embedded)          | ef=100 | 4.4s  |  7,363 | 1.000     |
| qdrant-client local (exact, py) | —      | 3.1s  |     36 | 1.000     |

Takeaways: ~50% of FAISS's single-thread HNSW throughput at equal recall
(FAISS has hand-written AVX2/AVX512 distance kernels; vecdb relies on compiler
auto-vectorization), slightly ahead of Chroma's full embedded stack, and the
qdrant number reflects its in-process Python prototyping mode, not a server.

## Python bindings

```python
from pyvecdb import VecDB
db = VecDB(dim=128, M=16, ef_construction=200)
db.add(vectors)                      # (n, dim) float32, batch
ids, dists = db.search(queries, k=10, ef=100)
db.save("index.vecdb"); db = VecDB.load("index.vecdb")
```

Build the shared lib with `gcc -O3 -march=native -fPIC -shared vecdb.c -o libvecdb.so -lm`.

## TurboQuant compression (arXiv:2504.19874, ICLR 2026)

Implements the core MSE variant of Google's TurboQuant: norm separation,
randomized Hadamard rotation (3 rounds, O(d log d)), and per-coordinate
Lloyd-Max quantization against a single analytic N(0,1) codebook. Fully
data-oblivious: no training phase, no calibration data, online inserts.
The paper's 1-bit QJL residual stage (unbiased inner-product estimates)
is not implemented.

### Quantization shoot-out (20k x 128, single thread, flat scans over codes)

| method                    | codebook | B/vec | build | QPS   | recall@10 |
|---------------------------|----------|-------|-------|-------|-----------|
| fp32 exact                | —        | 512   | —     | 2,170 | 1.000     |
| **TurboQuant 4-bit**      | none     | 76    | 0.2s  | 444   | 0.622     |
| **TurboQuant 4-bit +rerank** | none  | 76    | 0.2s  | 414   | **1.000** |
| **TurboQuant 8-bit**      | none     | 140   | 0.3s  | 240   | 0.961     |
| faiss SQ4                 | min/max  | 64    | 0.0s  | 1,380 | 0.628     |
| faiss SQ8                 | min/max  | 128   | 0.0s  | 1,988 | 0.966     |
| faiss PQ64x8              | trained  | 64    | 33.9s | 1,531 | 0.703     |

(+rerank = top-100 by compressed distance, exact re-score, top-10. All
+rerank variants reach 1.000.)

Reading: TurboQuant matches FAISS SQ4's quantization quality at the same
bit-width with zero training (PQ edges both on raw recall but costs 34s of
k-means). The QPS gap vs FAISS is the scan kernel, not the algorithm —
FAISS scans codes with hand-written SIMD; ours is scalar C.

## QJL residual stage (TurboQuant-prod, Algorithm 2)

The full inner-product variant is now implemented: 1-bit Quantized
Johnson-Lindenstrauss on the quantization residual — store sign(S·r) (packed,
pdim bits) plus ||r||, estimate <q,x> as <q,x_hat> + ||r||·sqrt(pi/2)/d·<Sq, signs>.
S·q is computed once per query; the per-vector sign-dot uses AVX-512 mask
registers (stored bits ARE the __mmask16).

### Validation: unbiasedness (unit vectors, est. vs true inner products)

| variant             | bias (mean err) | RMSE    |
|---------------------|-----------------|---------|
| TQ-mse 4-bit        | +0.00237        | 0.01386 |
| TQ-prod 4-bit + QJL | -0.00016        | 0.01043 |
| TQ-mse 8-bit        | -0.00001        | 0.00090 |
| TQ-prod 8-bit + QJL | -0.00000        | 0.00075 |

QJL removes the bias (~15x at 4-bit) and improves marginal RMSE, exactly as
Theorem 2 predicts.

### ...but it is NOT a recall booster (and that's instructive)

| method                  | B/vec | QPS   | recall@10 |
|-------------------------|-------|-------|-----------|
| TQ 4-bit                | 80    | 2,235 | 0.622     |
| TQ 4-bit + QJL          | 100   | 1,345 | 0.223     |
| TQ 8-bit                | 144   | 467   | 0.961     |
| TQ 8-bit + QJL          | 164   | 395   | 0.866     |

The MSE-only estimator's error is *deterministic given the codes* and highly
correlated across similar candidates — for a fixed query, errors largely
cancel in the ranking. Its bias is also near-monotone (shrinkage), which
preserves order. QJL replaces that with *independent per-vector noise*: the
marginal RMSE is lower, but the candidate-differential noise is higher, which
is what ranking actually feels. At b >= 4 the MSE bias is already negligible,
so QJL costs recall.

Practical guidance: use the MSE variant (+rerank) for retrieval; use the QJL
variant where unbiased inner-product *values* matter — KV-cache attention
scores, distributed sketches, estimation pipelines (the paper's primary
targets). The paper's NN-recall wins over PQ are at very low bit-widths,
where the MSE bias (2/pi at b=1) genuinely distorts rankings.

## Optimization log: where the throughput came from

HNSW @ ef=50 (recall 1.000), 20k x 128, single thread, best-of-3:

| stage                          | QPS    | vs faiss HNSW |
|--------------------------------|--------|---------------|
| scalar (auto-vectorized)       | 10,481 | 50%           |
| + AVX-512 distance kernel      | 12,409 | 59%           |
| + interleaved prefetch         | 16,423 | 78%           |

Prefetching is the one-ahead pattern (hnswlib-style): while computing the
distance to neighbor j, issue prefetches for neighbor j+1's visited mark and
first two cache lines. A naive prefetch-all-then-compute pass *regressed*
small-ef workloads — prefetch distance must match compute latency.

TurboQuant 4-bit scan: 444 -> 2,235 QPS (5x) via the algebraic decomposition
(precomputed ||L[c]||^2 turns the scan into a dot product) + register-resident
codebook decode (_mm512_permutexvar_ps).

## v2 layout + blocked exact scan

Two structural changes:

1. **Flat link arenas.** Level-0 neighbor lists moved from per-node mallocs
   (double pointer-chase, heap-scattered) into one contiguous arena with
   fixed stride — a node's links live at `i * stride`, computable without a
   dependent load. Upper-level lists (only ~1/M of nodes have them) stay as
   one block per node.
2. **Blocked exact search.** `vecdb_search_flat_batch` processes 8 queries
   per pass using `||q-x||^2 = ||q||^2 - 2<q,x> + ||x||^2` with cached
   `||x||^2`: every vector load feeds 8 FMA chains, raising arithmetic
   intensity 8x and lifting the scan off the memory-bandwidth wall.

### Final numbers (20k x 128, single thread, best-of-3, recall verified)

| index               | QPS    | vs faiss          |
|---------------------|--------|-------------------|
| flat exact (batch)  |  3,928 | **1.9x faster** than faiss IndexFlatL2 (2,039) |
| hnsw ef=50  (r=1.0) | 16,578 | 79%               |
| hnsw ef=100 (r=1.0) | 12,727 | 83%               |
| hnsw ef=200 (r=1.0) |  8,574 | 91%               |
| build               |  3.5s  | faiss 3.2s        |

Cumulative HNSW progression @ ef=100: 8,154 (scalar) -> 10,652 (AVX-512)
-> 11,411 (prefetch) -> 12,727 (link arena). The flat-scan win over FAISS
holds for this batch size (200 queries) and machine, single-threaded;
FAISS's BLAS-backed path may behave differently at other shapes.
