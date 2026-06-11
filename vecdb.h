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

size_t vecdb_count(const VecDB *db);
int    vecdb_dim(const VecDB *db);

/* Exact search. Fills out[0..k-1], returns number found. */
int vecdb_search_flat(const VecDB *db, const float *query, int k, VecResult *out);

/* Blocked exact search over nq queries at once (out has nq*k slots).
 * Returns per-query result count. Much faster than per-query calls. */
int vecdb_search_flat_batch(const VecDB *db, const float *queries, int nq,
                            int k, VecResult *out);

/* Approximate search via HNSW. ef >= k controls accuracy/speed. */
int vecdb_search_hnsw(const VecDB *db, const float *query, int k, int ef, VecResult *out);

/* Binary persistence (vectors + graph). Returns 0 on success. */
int    vecdb_save(const VecDB *db, const char *path);
VecDB *vecdb_load(const char *path);

#endif /* VECDB_H */
