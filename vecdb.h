/* vecdb.h — minimal vector database in C
 *
 * Two indexes over the same storage:
 *   - Flat:  exact brute-force scan (ground truth, small datasets)
 *   - HNSW:  approximate nearest neighbor graph (Malkov & Yashunin, 2016)
 *
 * Distances: squared L2. (Cosine = L2 on normalized vectors.)
 */
#ifndef VECDB_H
#define VECDB_H

#include <stdint.h>
#include <stddef.h>

typedef struct VecDB VecDB;

typedef struct {
    uint64_t id;     /* user-supplied id */
    float    dist;   /* squared L2 distance to query */
} VecResult;

typedef struct {
    int    dim;              /* vector dimensionality */
    int    M;                /* HNSW: max links per node on upper layers (level0 uses 2M) */
    int    ef_construction;  /* HNSW: beam width during insert */
    size_t initial_capacity; /* preallocated slots (grows automatically) */
    uint64_t seed;           /* RNG seed for level sampling */
} VecDBConfig;

VecDBConfig vecdb_default_config(int dim);

VecDB *vecdb_create(const VecDBConfig *cfg);
void   vecdb_free(VecDB *db);

/* Insert one vector. Returns internal index (>=0) or -1 on error. */
int64_t vecdb_add(VecDB *db, uint64_t id, const float *vec);
/* Bulk‑insert a whole batch of vectors in a single C call.
 * The function copies the data, updates the ID map and wires the HNSW
 * graph exactly as vecdb_add() does, but without the Python‑level loop.
 * Returns 0 on success, -1 on allocation error or duplicate ID.
 */
int vecdb_add_bulk(VecDB *db,
                    const uint64_t *ids,
                    const float *vecs,
                    size_t n);

/* Bulk insert with a parallel storage phase (OpenMP). Graph construction
 * stays serial, so the resulting index is identical to vecdb_add_bulk;
 * the parallel win is the per-vector norm computation. threads<=0 uses the
 * OpenMP default. Without OpenMP this is identical to vecdb_add_bulk. */
int vecdb_add_bulk_mt(VecDB *db, const uint64_t *ids, const float *vecs,
                      size_t n, int threads);

/* Delete a vector by its user-supplied id (O(1), tombstone). The node keeps
 * carrying graph connectivity but is excluded from all results.
 * Returns 0 on success, -1 if id not found. */
int vecdb_delete(VecDB *db, uint64_t id);

/* Rebuild the index without tombstoned nodes (call after many deletes).
 * Returns 0 on success. */
int vecdb_compact(VecDB *db);

size_t vecdb_count(const VecDB *db);
int    vecdb_dim(const VecDB *db);

/* Exact search. Fills out[0..k-1], returns number found. */
int vecdb_search_flat(const VecDB *db, const float *query, int k, VecResult *out);

/* Blocked exact search over nq queries at once (out has nq*k slots).
 * Returns per-query result count. Much faster than per-query calls. */
int vecdb_search_flat_batch(const VecDB *db, const float *queries, int nq,
                            int k, VecResult *out);

/* Approximate search via HNSW. ef >= k controls accuracy/speed.
 * Thread-safe: any number of threads may search concurrently; writes
 * (add/delete/compact) require external exclusion from searches. */
int vecdb_search_hnsw(const VecDB *db, const float *query, int k, int ef, VecResult *out);

/* ---- filtered search ----
 * A mask is a byte per internal slot (vecdb_slots() bytes), 1 = allowed.
 * Build one from user ids with vecdb_make_mask (mode 0 allow / 1 deny;
 * returns the number of ids resolved, unknown ids are ignored).
 * HNSW traverses through disallowed nodes (filters cannot disconnect the
 * search) but excludes them from results. For very selective filters
 * (<~1% allowed) prefer the exact filtered scan or raise ef. */
/* Batched HNSW search (parallel over queries under OpenMP). mask may be
 * NULL; out has nq*k slots, unfilled slots id=UINT64_MAX/dist=inf. */
int vecdb_search_hnsw_batch(const VecDB *db, const float *queries, int nq,
                            int k, int ef, const uint8_t *mask, VecResult *out);

size_t  vecdb_slots(const VecDB *db);

/* Set OpenMP thread count for parallel batch searches (<=0 = default). */
void    vecdb_set_threads(int n);
int64_t vecdb_make_mask(const VecDB *db, const uint64_t *ids, size_t n,
                        int mode, uint8_t *mask);
int vecdb_search_hnsw_filtered(const VecDB *db, const float *query, int k, int ef,
                               const uint8_t *mask, VecResult *out);
int vecdb_search_flat_batch_filtered(const VecDB *db, const float *queries,
                                     int nq, int k, const uint8_t *mask,
                                     VecResult *out);

/* Binary persistence (vectors + graph). Returns 0 on success. */
int    vecdb_save(const VecDB *db, const char *path);
VecDB *vecdb_load(const char *path);

#endif /* VECDB_H */
