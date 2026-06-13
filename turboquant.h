/* turboquant.h — TurboQuant-style quantized vector index.
 *
 * Implements the core MSE variant of TurboQuant (Zandieh et al., 2025,
 * arXiv:2504.19874): norm separation + random orthogonal rotation
 * (randomized Hadamard, 3 rounds) + per-coordinate Lloyd-Max quantization
 * against the standard normal, one shared data-oblivious codebook.
 * The paper's 1-bit QJL residual stage (unbiased inner products) is not
 * implemented here.
 *
 * Storage per vector: dim*bits/8 code bytes + 4-byte norm + 8-byte id.
 * Search: asymmetric — query stays float, database vectors stay quantized.
 */
#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>
#include "vecdb.h"   /* VecResult */

typedef struct TQIndex TQIndex;

/* bits: 4 or 8 code bits per coordinate. */
TQIndex *tq_create(int dim, int bits, uint64_t seed);
/* use_qjl: adds the paper's 1-bit QJL residual stage (+1 effective bit per
 * coordinate + 4 bytes for the residual norm) for unbiased inner products. */
TQIndex *tq_create2(int dim, int bits, int use_qjl, uint64_t seed);
void     tq_free(TQIndex *tq);

int64_t  tq_add(TQIndex *tq, uint64_t id, const float *vec);
size_t   tq_count(const TQIndex *tq);

/* Brute-force scan over quantized codes. Returns number of results. */
int tq_search(const TQIndex *tq, const float *query, int k, VecResult *out);

/* Batched scan, parallel over queries under OpenMP. */
int tq_search_batch(const TQIndex *tq, const float *queries, int nq, int k,
                    VecResult *out);

/* Bytes used by codes+norms+ids (excludes struct overhead). */
size_t tq_memory_bytes(const TQIndex *tq);

#endif /* TURBOQUANT_H */

/* ---- codec API for the hybrid HNSW-over-codes index ----
 * tq_encode stores a vector's code (== tq_add). tq_query_begin builds a
 * reusable per-query context; tq_score_code returns the estimated squared
 * L2 distance from that query to stored code `slot`; tq_query_free releases
 * it. These let an external graph traverse on TurboQuant code distances. */
typedef struct TQQuery TQQuery;
int64_t  tq_encode(TQIndex *tq, uint64_t id, const float *vec);
TQQuery *tq_query_begin(const TQIndex *tq, const float *query);
float    tq_score_code(const TQQuery *q, uint32_t slot);
void     tq_query_free(TQQuery *q);
