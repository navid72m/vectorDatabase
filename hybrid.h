/* hybrid.h — HNSW-over-TurboQuant-codes index with fp32 rerank.
 *
 * Design A: the HNSW graph traverses on TurboQuant code distances, so the
 * hot path touches ~bits/32 of the memory an fp32 graph would.
 * Design B: the top candidates are reranked by exact fp32 distance, which
 * recovers the recall the quantized traversal gives up.
 */
#ifndef HYBRID_H
#define HYBRID_H

#include <stdint.h>
#include <stddef.h>
#include "vecdb.h"   /* VecResult */

typedef struct HybridIndex HybridIndex;

typedef struct {
    int    dim;
    int    M;
    int    ef_construction;
    int    bits;             /* 4 or 8: TurboQuant code width           */
    int    qjl;              /* enable QJL residual stage               */
    int    rerank_mult;      /* rerank top (rerank_mult*k); 0 => default 4 */
    size_t initial_capacity;
    uint64_t seed;
} HybridConfig;

HybridIndex *hybrid_create(const HybridConfig *cfg);
void         hybrid_free(HybridIndex *h);

int64_t hybrid_add(HybridIndex *h, uint64_t id, const float *vec);
int     hybrid_search(const HybridIndex *h, const float *query, int k, int ef,
                      VecResult *out);

size_t  hybrid_count(const HybridIndex *h);
int     hybrid_dim(const HybridIndex *h);

/* Total bytes; include_fp32=0 reports the code+graph footprint that a
 * production (mmap-fp32) deployment would keep resident. */
size_t  hybrid_memory_bytes(const HybridIndex *h, int include_fp32);

#endif /* HYBRID_H */
