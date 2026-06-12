# vecdb — a small vector database in C

`vecdb` is a dependency-light vector index implemented in C11 with Python bindings through `ctypes`. It stores fixed-size `float32` vectors and supports exact and approximate nearest-neighbor search using squared L2 distance.

## What it implements

- **Flat exact search** — brute-force L2 scan.
- **Blocked exact batch search** — processes up to 8 queries per pass so one vector load feeds multiple query accumulators.
- **HNSW approximate search** — graph index using greedy descent, beam search (`ef`), and heuristic neighbor selection.
- **Binary persistence** — save/load an index with vectors, IDs, levels, and neighbor links.
- **Delete API** — delete vectors by user ID. Deletion is linear-time and keeps storage dense by swapping with the last vector.
- **TurboQuant compressed index** — optional 4-bit or 8-bit compressed brute-force index with randomized Hadamard rotation, norm separation, Lloyd-Max Gaussian quantization, and optional QJL residual estimation.

The C API is declared in `vecdb.h`. The Python API lives in `pyvecdb.py` and loads `libvecdb.so` from the project directory by default.

## Build

Requires `clang` and `numpy`.

```sh
make clean
make
```

This builds:

- `libvecdb.so` — shared C library containing `vecdb` and TurboQuant symbols.
- `bench` — small smoke/demo program.

Generated artifacts are ignored by `.gitignore`.

## Quick start

```sh
./bench 50000 128
```

`bench` generates random vectors, inserts them into an HNSW index, runs one approximate query, prints the top result, and writes `index.vecdb`.

## Python usage

```python
import numpy as np
from pyvecdb import VecDB, TQIndex

vectors = np.random.rand(10_000, 128).astype(np.float32)
ids = np.arange(vectors.shape[0], dtype=np.uint64)

# VecDB: exact flat and approximate HNSW search
db = VecDB(dim=128, M=16, ef_construction=200)
db.add(vectors, ids)

queries = vectors[:5]
flat_ids, flat_dists = db.search(queries, k=10, ef=100, exact=True)
hnsw_ids, hnsw_dists = db.search(queries, k=10, ef=100, exact=False)

# Delete by user ID. This is O(N) per ID and swaps the removed vector with the last one.
db.delete(np.array([42], dtype=np.uint64))

# Persist and reload
db.save("index.vecdb")
loaded = VecDB.load("index.vecdb")

# TurboQuant: compressed brute-force index, no training phase
tq = TQIndex(dim=128, bits=4, qjl=False)
tq.add(vectors, ids)
tq_ids, tq_dists = tq.search(queries, k=10)
print("TurboQuant memory bytes:", tq.memory_bytes())
```

`VecDB.search()` returns `(ids, distances)`, both shaped `(n_queries, k)`. Missing result slots are filled with `id = 2**64 - 1` and `dist = inf`. Distances are squared L2 values.

## C API

```c
#include "vecdb.h"

VecDBConfig cfg = vecdb_default_config(128);
cfg.M = 16;
cfg.ef_construction = 200;
cfg.initial_capacity = 1024;

VecDB *db = vecdb_create(&cfg);

float vec[128]; // fill with your data
vecdb_add(db, 42, vec);

VecResult out[10];
vecdb_search_flat(db, query, 10, out);
vecdb_search_hnsw(db, query, 10, 100, out);

vecdb_delete(db, 42);

vecdb_save(db, "index.vecdb");
VecDB *loaded = vecdb_load("index.vecdb");

vecdb_free(db);
vecdb_free(loaded);
```

Core C functions:

- `vecdb_default_config(dim)`
- `vecdb_create()`, `vecdb_free()`
- `vecdb_add(db, id, vector)`
- `vecdb_delete(db, id)`
- `vecdb_count()`, `vecdb_dim()`
- `vecdb_search_flat()`
- `vecdb_search_flat_batch()`
- `vecdb_search_hnsw()`
- `vecdb_save()`, `vecdb_load()`

## TurboQuant details

`TQIndex` is a compressed brute-force index, not an HNSW graph.

Per-vector encoding pipeline:

1. Store the original norm.
2. Normalize the vector direction.
3. Pad dimension to the next power of two.
4. Apply three rounds of sign flips plus fast Walsh-Hadamard transform.
5. Quantize each rotated coordinate with an analytic Lloyd-Max Gaussian codebook.
6. For 8-bit mode, snap reconstructions to signed int8 and use an AVX512-VNNI scan path when available.
7. Optional QJL mode stores residual sign bits and residual norms for an unbiased inner-product estimator.

Supported Python constructor:

```python
tq = TQIndex(dim=128, bits=4, qjl=False, seed=123)
```

`bits` must be `4` or `8`. `qjl=True` enables the TurboQuant-prod residual stage. TurboQuant indexes currently expose `add`, `search`, `memory_bytes`, and `__len__`; they do not expose delete or persistence.

## Implementation notes

- Distance metric: squared L2. Cosine search can be done by normalizing vectors before insertion and querying.
- HNSW uses deterministic seeded `xorshift64*` level sampling.
- HNSW search uses epoch-tagged visited marks to avoid clearing a visited array per query.
- Flat batch search uses the identity `||q - x||^2 = ||q||^2 - 2<q,x> + ||x||^2` with cached vector norms.
- `vecdb.c` has an AVX-512 distance kernel when compiled with AVX-512; otherwise it falls back to scalar code that is intended to auto-vectorize under `-O3`.
- `turboquant.c` has AVX-512 4-bit lookup and VNNI 8-bit scan paths when the compiler target supports them; otherwise it falls back to scalar scans.
- The Makefile uses `-march=native`, so build artifacts are machine-specific.

## Known limits

- Single-threaded indexing and search.
- HNSW deletion removes the node by swap-with-last instead of repairing the graph. Rebuild after many deletes if graph quality matters.
- Python `VecDB.delete()` is O(N) per ID and invalidates external assumptions about internal node order.
- TurboQuant is a brute-force compressed index; it does not build an HNSW graph.
- TurboQuant has no delete or save/load API yet.
- Persistence format is versioned but not a long-term stable external format.

## Project layout

```text
vecdb.h        # C API
vecdb.c        # VecDB storage, flat search, HNSW, persistence
turboquant.c   # TurboQuant compressed index
pyvecdb.py     # ctypes Python bindings
bench.c        # small random-data smoke/demo program
Makefile       # build rules for libvecdb.so and bench
pyproject.toml # Python package metadata
```

## License

MIT.
