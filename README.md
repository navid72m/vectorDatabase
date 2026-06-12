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

## Build from source

Requirements:

- `clang`
- `numpy` for the Python bindings
- A Unix-like shell with `make`

```sh
make clean
make
```

`make` builds `libvecdb.so`. Build the demo program separately:

```sh
make bench
```

Useful targets:

| Target | Result |
|--------|--------|
| `make` / `make all` | Builds `libvecdb.so` |
| `make bench` | Builds the `bench` demo executable |
| `make clean` | Removes `.o`, `.so`, and `bench` build artifacts |

The Makefile uses `-O3 -march=native`, so generated binaries are machine-specific. If you need to override the compiler or flags:

```sh
make clean
make CC=clang CFLAGS="-O3 -Wall -Wextra -fPIC"
```

For local Python use, run scripts from the repo root or set `PYTHONPATH=.`:

```sh
PYTHONPATH=. python script.py
```

`pyvecdb.py` loads `libvecdb.so` from the same directory as the Python file. To load a library from somewhere else, set `VECDB_LIB`:

```sh
VECDB_LIB=/path/to/libvecdb.so PYTHONPATH=. python script.py
```

## Quick start

```sh
make
make bench
./bench 50000 128
```

`bench` generates random vectors, inserts them into an HNSW index, runs one approximate query, prints the top result, and writes `index.vecdb`.

## Python API

`pyvecdb.py` exposes two index classes:

| Class | Purpose |
|-------|---------|
| `VecDB` | Full vector index with flat exact search, HNSW approximate search, delete, save, and load |
| `TQIndex` | TurboQuant compressed brute-force index |

### `VecDB`

```python
from pyvecdb import VecDB

db = VecDB(
    dim=128,
    M=16,
    ef_construction=200,
    capacity=1024,
    seed=0x9E3779B97F4A7C15,
)
```

Constructor arguments:

| Argument | Meaning |
|----------|---------|
| `dim` | Vector dimensionality |
| `M` | HNSW max links per node on upper layers; level 0 uses `2 * M` |
| `ef_construction` | HNSW beam width during inserts |
| `capacity` | Initial vector capacity; the index grows automatically |
| `seed` | Deterministic RNG seed for HNSW level sampling |

Core methods:

```python
db.add(vectors, ids=None)          # returns None; ids default to current_count..current_count+n
db.delete(ids)                     # raises RuntimeError if any id is missing
ids, distances = db.search(queries, k=10, ef=100, exact=False)
db.save(path)
loaded = VecDB.load(path)
len(db)                            # number of stored vectors
db.dim                             # vector dimensionality
```

Input requirements:

- `vectors` and `queries` must be convertible to contiguous `float32` arrays.
- Shape must be `(n, dim)`; a 1D vector is accepted as `(1, dim)`.
- `ids`, when provided, must be shape `(n,)`.
- `ef` is clamped up to `k` if `ef < k`.

`search()` returns `(ids, distances)`, both shaped `(n_queries, k)`. Missing result slots are filled with `id = 2**64 - 1` and `dist = inf`. Distances are squared L2 values.

Example:

```python
import numpy as np
from pyvecdb import VecDB

vectors = np.random.rand(10_000, 128).astype(np.float32)
ids = np.arange(vectors.shape[0], dtype=np.uint64)

db = VecDB(dim=128, M=16, ef_construction=200)
db.add(vectors, ids)

queries = vectors[:5]
flat_ids, flat_dists = db.search(queries, k=10, ef=100, exact=True)
hnsw_ids, hnsw_dists = db.search(queries, k=10, ef=100, exact=False)

db.delete(np.array([42], dtype=np.uint64))

db.save("index.vecdb")
loaded = VecDB.load("index.vecdb")
```

Delete caveat: `VecDB.delete()` is O(N) per ID and keeps storage dense by swapping the removed vector with the last vector. External references to internal node positions become invalid after deletion. After many deletes, rebuilding the index can improve HNSW graph quality.

### `TQIndex`

```python
from pyvecdb import TQIndex

tq = TQIndex(dim=128, bits=4, qjl=False, seed=123)
```

Constructor arguments:

| Argument | Meaning |
|----------|---------|
| `dim` | Vector dimensionality; internally padded to the next power of two |
| `bits` | `4` or `8` bits per coordinate |
| `qjl` | Enable TurboQuant-prod residual estimation |
| `seed` | Deterministic RNG seed for Hadamard signs and QJL projection |

Core methods:

```python
tq.add(vectors, ids=None)
ids, distances = tq.search(queries, k=10)
bytes_used = tq.memory_bytes()
len(tq)
```

TurboQuant is a compressed brute-force index, not an HNSW graph. It currently exposes `add`, `search`, `memory_bytes`, and `__len__`; it does not expose delete or persistence.

Example:

```python
import numpy as np
from pyvecdb import TQIndex

vectors = np.random.rand(10_000, 128).astype(np.float32)
ids = np.arange(vectors.shape[0], dtype=np.uint64)

tq = TQIndex(dim=128, bits=4, qjl=False)
tq.add(vectors, ids)

queries = vectors[:5]
tq_ids, tq_dists = tq.search(queries, k=10)
print(tq.memory_bytes())
```

## C API

`vecdb.h` declares the public C interface.

### Types

```c
typedef struct VecDB VecDB;

typedef struct {
    uint64_t id;     /* user-supplied id */
    float    dist;   /* squared L2 distance to query */
} VecResult;

typedef struct {
    int    dim;
    int    M;
    int    ef_construction;
    size_t initial_capacity;
    uint64_t seed;
} VecDBConfig;
```

Return semantics:

- `vecdb_create()` returns `NULL` on invalid config or allocation failure.
- `vecdb_add()` returns the internal index on success, or `-1` on failure.
- `vecdb_delete()` returns `0` on success, or `-1` if the ID is not found.
- `vecdb_search_*()` returns the number of results written.
- `vecdb_save()` returns `0` on success, or `-1` on failure.
- `vecdb_load()` returns `NULL` on open/read/format failure.

### Example

```c
#include "vecdb.h"

int main(void) {
    VecDBConfig cfg = vecdb_default_config(128);
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.initial_capacity = 1024;
    cfg.seed = 0x9E3779B97F4A7C15ULL;

    VecDB *db = vecdb_create(&cfg);
    if (!db) return 1;

    float vec[128]; // fill with your data
    vecdb_add(db, 42, vec);

    float query[128]; // fill with your data
    VecResult out[10];

    int n_flat = vecdb_search_flat(db, query, 10, out);
    int n_hnsw = vecdb_search_hnsw(db, query, 10, 100, out);

    vecdb_delete(db, 42);

    if (vecdb_save(db, "index.vecdb") != 0) return 1;

    VecDB *loaded = vecdb_load("index.vecdb");

    vecdb_free(db);
    vecdb_free(loaded);
    return 0;
}
```

Core C functions:

| Function | Purpose |
|----------|---------|
| `vecdb_default_config(dim)` | Fill a config with defaults: `M=16`, `ef_construction=200`, `initial_capacity=1024` |
| `vecdb_create(&cfg)` | Allocate and initialize a database |
| `vecdb_free(db)` | Free all memory |
| `vecdb_add(db, id, vector)` | Insert one vector |
| `vecdb_delete(db, id)` | Delete one vector by user ID |
| `vecdb_count(db)` | Number of stored vectors |
| `vecdb_dim(db)` | Vector dimensionality |
| `vecdb_search_flat(db, query, k, out)` | Exact single-query search |
| `vecdb_search_flat_batch(db, queries, nq, k, out)` | Exact batched search over up to 8 queries per pass |
| `vecdb_search_hnsw(db, query, k, ef, out)` | Approximate HNSW search |
| `vecdb_save(db, path)` | Persist vectors and graph |
| `vecdb_load(path)` | Load a persisted index |

## TurboQuant details

`TQIndex` encodes each vector as a compact code and searches by scanning those codes.

Per-vector encoding pipeline:

1. Store the original norm.
2. Normalize the vector direction.
3. Pad dimension to the next power of two.
4. Apply three rounds of sign flips plus fast Walsh-Hadamard transform.
5. Quantize each rotated coordinate with an analytic Lloyd-Max Gaussian codebook.
6. For 8-bit mode, snap reconstructions to signed int8 and use an AVX512-VNNI scan path when available.
7. Optional QJL mode stores residual sign bits and residual norms for an unbiased inner-product estimator.

Distances returned by `TQIndex.search()` are estimated squared L2 distances for the compressed representation.

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
