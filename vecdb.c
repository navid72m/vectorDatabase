/* vecdb.c — implementation. See vecdb.h for API.
 *
 * Memory layout (v2):
 *   - vectors: one dense row-major array (good for blocked flat scans)
 *   - level-0 links: one flat arena, fixed stride (M0+2) u32 per node:
 *     [count, n0, n1, ...]. Address = i * stride — no pointer chase.
 *   - upper-level links: one malloc per node that has them (~1/M of nodes),
 *     levels 1..L contiguous with stride (M+2) u32.
 */
#include "vecdb.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Distance kernel: squared L2.                                        */
/* AVX-512 path: 2x unrolled 16-wide FMA + masked tail.                */
/* Fallback: 4-way unrolled scalar that -O3 auto-vectorizes.           */
/* ------------------------------------------------------------------ */
#if defined(__AVX512F__)
#include <immintrin.h>
#define VECDB_AVX512 1
static float l2sq(const float *restrict a, const float *restrict b, int dim)
{
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i),
                                  _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16),
                                  _mm512_loadu_ps(b + i + 16));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 16 <= dim; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i),
                                 _mm512_loadu_ps(b + i));
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    if (i < dim) {
        __mmask16 m = (__mmask16)((1u << (dim - i)) - 1u);
        __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + i),
                                 _mm512_maskz_loadu_ps(m, b + i));
        s += _mm512_reduce_add_ps(_mm512_mul_ps(d, d));
    }
    return s;
}
#else
static float l2sq(const float *restrict a, const float *restrict b, int dim)
{
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float d0 = a[i+0] - b[i+0];
        float d1 = a[i+1] - b[i+1];
        float d2 = a[i+2] - b[i+2];
        float d3 = a[i+3] - b[i+3];
        s0 += d0*d0; s1 += d1*d1; s2 += d2*d2; s3 += d3*d3;
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < dim; i++) { float d = a[i] - b[i]; s += d*d; }
    return s;
}
#endif

/* ------------------------------------------------------------------ */
/* Binary heap of (dist, node) — used as both min- and max-heap        */
/* ------------------------------------------------------------------ */
typedef struct { float dist; uint32_t node; } HItem;

typedef struct {
    HItem *a;
    int    n, cap;
    int    is_max;          /* 1: largest dist on top, 0: smallest on top */
} Heap;

static int heap_before(const Heap *h, const HItem *x, const HItem *y)
{
    return h->is_max ? (x->dist > y->dist) : (x->dist < y->dist);
}

static void heap_init(Heap *h, int cap, int is_max)
{
    h->a = malloc((size_t)cap * sizeof(HItem));
    h->n = 0; h->cap = cap; h->is_max = is_max;
}

static void heap_destroy(Heap *h) { free(h->a); h->a = NULL; h->n = h->cap = 0; }

static void heap_push(Heap *h, float dist, uint32_t node)
{
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 16;
        h->a = realloc(h->a, (size_t)h->cap * sizeof(HItem));
    }
    int i = h->n++;
    h->a[i] = (HItem){dist, node};
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!heap_before(h, &h->a[i], &h->a[p])) break;
        HItem t = h->a[i]; h->a[i] = h->a[p]; h->a[p] = t;
        i = p;
    }
}

static HItem heap_pop(Heap *h)
{
    HItem top = h->a[0];
    h->a[0] = h->a[--h->n];
    int i = 0;
    for (;;) {
        int l = 2*i + 1, r = 2*i + 2, best = i;
        if (l < h->n && heap_before(h, &h->a[l], &h->a[best])) best = l;
        if (r < h->n && heap_before(h, &h->a[r], &h->a[best])) best = r;
        if (best == i) break;
        HItem t = h->a[i]; h->a[i] = h->a[best]; h->a[best] = t;
        i = best;
    }
    return top;
}

/* ------------------------------------------------------------------ */
struct VecDB {
    /* config */
    int dim, M, M0, ef_construction;
    double mult;                 /* 1/ln(M), level sampling           */
    uint64_t rng;

    /* storage */
    float    *vecs;              /* count * dim, row-major            */
    float    *vnorms;            /* count: squared L2 norms           */
    uint64_t *ids;
    size_t    count, cap;

    /* graph */
    uint32_t  *l0;               /* cap * l0_stride: level-0 arena    */
    size_t     l0_stride;        /* M0 + 2 u32 per node               */
    uint32_t **upper;            /* per-node upper-level block or NULL */
    uint8_t   *node_level;
    int64_t    entry;            /* entry point node, -1 if empty     */
    int        max_level;

    /* tombstone deletes: dead nodes keep carrying graph connectivity
     * (traversal passes through them) but are filtered from results.   */
    uint8_t  *dead;              /* cap bytes, 1 = tombstoned          */
    size_t    alive;             /* live vector count                  */

    /* id -> slot open-addressing hash map (O(1) delete/lookup)        */
    uint64_t *map_keys;
    uint32_t *map_vals;
    uint8_t  *map_state;         /* 0 empty, 1 used, 2 erased          */
    size_t    map_cap;           /* power of two                       */
    size_t    map_used;

    /* per-query visited marks (epoch trick, no memset per query)     */
    uint32_t *visited;
    uint32_t  epoch;
};

/* splitmix64 finalizer */
static uint64_t hash_u64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static void map_put(VecDB *db, uint64_t key, uint32_t val);

static void map_rehash(VecDB *db, size_t ncap)
{
    uint64_t *ok = db->map_keys; uint32_t *ov = db->map_vals;
    uint8_t *os = db->map_state; size_t oc = db->map_cap;
    db->map_keys  = malloc(ncap * sizeof(uint64_t));
    db->map_vals  = malloc(ncap * sizeof(uint32_t));
    db->map_state = calloc(ncap, 1);
    db->map_cap = ncap; db->map_used = 0;
    for (size_t i = 0; i < oc; i++)
        if (os[i] == 1) map_put(db, ok[i], ov[i]);
    free(ok); free(ov); free(os);
}

static void map_put(VecDB *db, uint64_t key, uint32_t val)
{
    if ((db->map_used + 1) * 10 >= db->map_cap * 7)       /* <70% load */
        map_rehash(db, db->map_cap ? db->map_cap * 2 : 2048);
    size_t m = db->map_cap - 1, i = hash_u64(key) & m;
    size_t grave = (size_t)-1;
    for (;;) {
        if (db->map_state[i] == 0) break;
        if (db->map_state[i] == 2) { if (grave == (size_t)-1) grave = i; }
        else if (db->map_keys[i] == key) { db->map_vals[i] = val; return; }
        i = (i + 1) & m;
    }
    if (grave != (size_t)-1) i = grave;
    db->map_keys[i] = key; db->map_vals[i] = val; db->map_state[i] = 1;
    db->map_used++;
}

/* returns slot or UINT32_MAX */
static uint32_t map_get(const VecDB *db, uint64_t key)
{
    if (!db->map_cap) return UINT32_MAX;
    size_t m = db->map_cap - 1, i = hash_u64(key) & m;
    for (;;) {
        if (db->map_state[i] == 0) return UINT32_MAX;
        if (db->map_state[i] == 1 && db->map_keys[i] == key) return db->map_vals[i];
        i = (i + 1) & m;
    }
}

static void map_del(VecDB *db, uint64_t key)
{
    if (!db->map_cap) return;
    size_t m = db->map_cap - 1, i = hash_u64(key) & m;
    for (;;) {
        if (db->map_state[i] == 0) return;
        if (db->map_state[i] == 1 && db->map_keys[i] == key) {
            db->map_state[i] = 2;
            db->map_used--;
            return;
        }
        i = (i + 1) & m;
    }
}

static const float *vec_at(const VecDB *db, uint32_t i) { return db->vecs + (size_t)i * db->dim; }

static inline uint32_t *node_links(const VecDB *db, uint32_t i, int level)
{
    if (level == 0) return db->l0 + (size_t)i * db->l0_stride;
    return db->upper[i] + (size_t)(level - 1) * (size_t)(db->M + 2);
}

/* xorshift64* RNG — deterministic given seed */
static double rng_uniform(VecDB *db)
{
    uint64_t x = db->rng;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    db->rng = x;
    return (double)((x * 0x2545F4914F6CDD1DULL) >> 11) / (double)(1ULL << 53);
}

static int sample_level(VecDB *db)
{
    double u = rng_uniform(db);
    if (u < 1e-12) u = 1e-12;
    int l = (int)(-log(u) * db->mult);
    return l > 32 ? 32 : l;
}

VecDBConfig vecdb_default_config(int dim)
{
    VecDBConfig c;
    c.dim = dim;
    c.M = 16;
    c.ef_construction = 200;
    c.initial_capacity = 1024;
    c.seed = 0x9E3779B97F4A7C15ULL;
    return c;
}

VecDB *vecdb_create(const VecDBConfig *cfg)
{
    if (!cfg || cfg->dim <= 0 || cfg->M < 2) return NULL;
    VecDB *db = calloc(1, sizeof(VecDB));
    db->dim = cfg->dim;
    db->M = cfg->M;
    db->M0 = cfg->M * 2;
    db->ef_construction = cfg->ef_construction < cfg->M ? cfg->M : cfg->ef_construction;
    db->mult = 1.0 / log((double)db->M);
    db->rng = cfg->seed ? cfg->seed : 88172645463325252ULL;
    db->cap = cfg->initial_capacity ? cfg->initial_capacity : 1024;
    db->l0_stride = (size_t)db->M0 + 2;
    db->vecs    = malloc(db->cap * (size_t)db->dim * sizeof(float));
    db->vnorms  = malloc(db->cap * sizeof(float));
    db->ids     = malloc(db->cap * sizeof(uint64_t));
    db->l0      = malloc(db->cap * db->l0_stride * sizeof(uint32_t));
    db->upper   = calloc(db->cap, sizeof(uint32_t *));
    db->node_level = calloc(db->cap, 1);
    db->dead    = calloc(db->cap, 1);
    db->visited = calloc(db->cap, sizeof(uint32_t));
    db->entry = -1;
    db->max_level = -1;
    db->epoch = 0;
    return db;
}

void vecdb_free(VecDB *db)
{
    if (!db) return;
    for (size_t i = 0; i < db->count; i++) free(db->upper[i]);
    free(db->upper); free(db->node_level); free(db->l0); free(db->dead);
    free(db->map_keys); free(db->map_vals); free(db->map_state);
    free(db->vecs); free(db->vnorms); free(db->ids); free(db->visited);
    free(db);
}

size_t vecdb_count(const VecDB *db) { return db->alive; }
int    vecdb_dim(const VecDB *db)   { return db->dim; }

static void grow(VecDB *db)
{
    size_t ncap = db->cap * 2;
    db->vecs    = realloc(db->vecs,   ncap * (size_t)db->dim * sizeof(float));
    db->vnorms  = realloc(db->vnorms, ncap * sizeof(float));
    db->ids     = realloc(db->ids,    ncap * sizeof(uint64_t));
    db->l0      = realloc(db->l0,     ncap * db->l0_stride * sizeof(uint32_t));
    db->upper   = realloc(db->upper,  ncap * sizeof(uint32_t *));
    db->node_level = realloc(db->node_level, ncap);
    db->dead    = realloc(db->dead, ncap);
    db->visited = realloc(db->visited, ncap * sizeof(uint32_t));
    memset(db->upper + db->cap, 0, (ncap - db->cap) * sizeof(uint32_t *));
    memset(db->node_level + db->cap, 0, ncap - db->cap);
    memset(db->dead + db->cap, 0, ncap - db->cap);
    memset(db->visited + db->cap, 0, (ncap - db->cap) * sizeof(uint32_t));
    db->cap = ncap;
}

static int link_cap(const VecDB *db, int level) { return level == 0 ? db->M0 : db->M; }

/* Greedy beam search on one layer. Results land in `res` (max-heap, size<=ef). */
static void search_layer(VecDB *db, const float *q, uint32_t entry, int ef,
                         int level, Heap *res, int filter_dead)
{
    db->epoch++;
    if (db->epoch == 0) {            /* wrapped: clear marks once */
        memset(db->visited, 0, db->cap * sizeof(uint32_t));
        db->epoch = 1;
    }

    Heap cand; heap_init(&cand, ef + 1, 0);   /* min-heap: closest first */

    float d0 = l2sq(q, vec_at(db, entry), db->dim);
    db->visited[entry] = db->epoch;
    heap_push(&cand, d0, entry);
    if (!filter_dead || !db->dead[entry]) heap_push(res, d0, entry);

    while (cand.n > 0) {
        HItem c = heap_pop(&cand);
        if (res->n >= ef && c.dist > res->a[0].dist) break;   /* res top = worst kept */

        const uint32_t *L = node_links(db, c.node, level);
        uint32_t n = L[0];
        if (n) {                                /* one-ahead prefetch pattern */
            __builtin_prefetch(&db->visited[L[1]], 0, 3);
            __builtin_prefetch(vec_at(db, L[1]), 0, 3);
        }
        for (uint32_t j = 1; j <= n; j++) {
            if (j < n) {                        /* prefetch next while computing */
                __builtin_prefetch(&db->visited[L[j+1]], 0, 3);
                const float *pn = vec_at(db, L[j+1]);
                __builtin_prefetch(pn, 0, 3);
                __builtin_prefetch((const char *)pn + 64, 0, 3);
            }
            uint32_t nb = L[j];
            if (db->visited[nb] == db->epoch) continue;
            db->visited[nb] = db->epoch;
            float d = l2sq(q, vec_at(db, nb), db->dim);
            if (res->n < ef || d < res->a[0].dist) {
                heap_push(&cand, d, nb);
                if (!filter_dead || !db->dead[nb]) {
                    heap_push(res, d, nb);
                    if (res->n > ef) heap_pop(res);
                }
            }
        }
    }
    heap_destroy(&cand);
}

/* Pure greedy descent (ef=1) used on layers above the target. */
static uint32_t greedy_step(VecDB *db, const float *q, uint32_t entry, int level)
{
    uint32_t cur = entry;
    float best = l2sq(q, vec_at(db, cur), db->dim);
    int improved = 1;
    while (improved) {
        improved = 0;
        const uint32_t *L = node_links(db, cur, level);
        uint32_t n = L[0];
        for (uint32_t j = 1; j <= n; j++)
            __builtin_prefetch(vec_at(db, L[j]), 0, 3);
        for (uint32_t j = 1; j <= n; j++) {
            float d = l2sq(q, vec_at(db, L[j]), db->dim);
            if (d < best) { best = d; cur = L[j]; improved = 1; }
        }
    }
    return cur;
}

/* Heuristic neighbor selection (Algorithm 4 in the HNSW paper). */
static int select_neighbors(VecDB *db, const float *base, HItem *cands, int ncand,
                            int M, uint32_t *out)
{
    for (int i = 1; i < ncand; i++) {
        HItem k = cands[i]; int j = i - 1;
        while (j >= 0 && cands[j].dist > k.dist) { cands[j+1] = cands[j]; j--; }
        cands[j+1] = k;
    }
    (void)base;
    int kept = 0;
    for (int i = 0; i < ncand && kept < M; i++) {
        const float *cv = vec_at(db, cands[i].node);
        int good = 1;
        for (int j = 0; j < kept; j++) {
            if (l2sq(cv, vec_at(db, out[j]), db->dim) < cands[i].dist) { good = 0; break; }
        }
        if (good) out[kept++] = cands[i].node;
    }
    for (int i = 0; i < ncand && kept < M; i++) {
        int dup = 0;
        for (int j = 0; j < kept; j++) if (out[j] == cands[i].node) { dup = 1; break; }
        if (!dup) out[kept++] = cands[i].node;
    }
    return kept;
}

/* Re-prune a node's neighbor list at `level` down to capacity. */
static void shrink_links(VecDB *db, uint32_t node, int level)
{
    int cap = link_cap(db, level);
    uint32_t *L = node_links(db, node, level);
    int n = (int)L[0];
    if (n <= cap) return;

    HItem *cands = malloc((size_t)n * sizeof(HItem));
    const float *base = vec_at(db, node);
    for (int i = 0; i < n; i++)
        cands[i] = (HItem){ l2sq(base, vec_at(db, L[1+i]), db->dim), L[1+i] };

    uint32_t *sel = malloc((size_t)cap * sizeof(uint32_t));
    int kept = select_neighbors(db, base, cands, n, cap, sel);
    L[0] = (uint32_t)kept;
    memcpy(&L[1], sel, (size_t)kept * sizeof(uint32_t));
    free(sel); free(cands);
}

int64_t vecdb_add(VecDB *db, uint64_t id, const float *vec)
{
    if (!db || !vec) return -1;
    if (db->count == db->cap) grow(db);

    if (map_get(db, id) != UINT32_MAX) return -1;   /* duplicate live id */

    uint32_t idx = (uint32_t)db->count++;
    db->dead[idx] = 0;
    db->alive++;
    map_put(db, id, idx);
    memcpy(db->vecs + (size_t)idx * db->dim, vec, (size_t)db->dim * sizeof(float));
    db->ids[idx] = id;
    {
        double nsq = 0.0;
        for (int i = 0; i < db->dim; i++) nsq += (double)vec[i] * vec[i];
        db->vnorms[idx] = (float)nsq;
    }

    int level = sample_level(db);
    db->node_level[idx] = (uint8_t)level;
    node_links(db, idx, 0)[0] = 0;
    if (level > 0) {
        db->upper[idx] = calloc((size_t)level * (size_t)(db->M + 2), sizeof(uint32_t));
    } else {
        db->upper[idx] = NULL;
    }

    if (db->entry < 0) {                 /* first element */
        db->entry = idx;
        db->max_level = level;
        return idx;
    }

    uint32_t ep = (uint32_t)db->entry;

    /* descend from the top to level+1 greedily */
    for (int l = db->max_level; l > level; l--)
        ep = greedy_step(db, vec, ep, l);

    /* insert on each layer from min(level, max_level) down to 0 */
    int start = level < db->max_level ? level : db->max_level;
    for (int l = start; l >= 0; l--) {
        Heap res; heap_init(&res, db->ef_construction + 1, 1);
        search_layer(db, vec, ep, db->ef_construction, l, &res, 0);

        int M = db->M;  /* links to create from the new node */
        uint32_t *sel = malloc((size_t)M * sizeof(uint32_t));
        int kept = select_neighbors(db, vec, res.a, res.n, M, sel);

        /* connect new -> selected */
        uint32_t *L = node_links(db, idx, l);
        for (int j = 0; j < kept; j++) L[1 + L[0]++] = sel[j];

        /* connect selected -> new, pruning overfull lists */
        for (int j = 0; j < kept; j++) {
            uint32_t *NL = node_links(db, sel[j], l);
            NL[1 + NL[0]++] = idx;
            if ((int)NL[0] > link_cap(db, l)) shrink_links(db, sel[j], l);
        }

        /* best candidate becomes entry for the next (lower) layer */
        float bd = 1e30f; uint32_t bn = ep;
        for (int j = 0; j < res.n; j++)
            if (res.a[j].dist < bd) { bd = res.a[j].dist; bn = res.a[j].node; }
        ep = bn;

        free(sel);
        heap_destroy(&res);
    }

    if (level > db->max_level) {
        db->max_level = level;
        db->entry = idx;
    }
    return idx;
}

/* ------------------------------------------------------------------ */
/* Delete (tombstones)                                                 */
/* A deleted node stays in the graph as a waypoint: traversal passes   */
/* through it (removing it would orphan regions it connects), but it   */
/* is excluded from results, scans, and persistence-rebuilt maps.      */
/* Slots are not reused; call vecdb_compact() to rebuild without the   */
/* dead nodes once they accumulate.                                    */
/* ------------------------------------------------------------------ */
int vecdb_delete(VecDB *db, uint64_t id)
{
    if (!db) return -1;
    uint32_t idx = map_get(db, id);
    if (idx == UINT32_MAX) return -1;
    db->dead[idx] = 1;
    db->alive--;
    map_del(db, id);
    return 0;
}

int vecdb_compact(VecDB *db)
{
    if (!db) return -1;
    if (db->alive == db->count) return 0;            /* nothing to do */
    VecDBConfig cfg;
    cfg.dim = db->dim; cfg.M = db->M;
    cfg.ef_construction = db->ef_construction;
    cfg.initial_capacity = db->alive ? db->alive : 1024;
    cfg.seed = db->rng;                              /* continue RNG stream */
    VecDB *nd = vecdb_create(&cfg);
    if (!nd) return -1;
    for (size_t i = 0; i < db->count; i++) {
        if (db->dead[i]) continue;
        if (vecdb_add(nd, db->ids[i], vec_at(db, (uint32_t)i)) < 0) {
            vecdb_free(nd);
            return -1;
        }
    }
    /* swap guts, free the old body */
    VecDB tmp = *db;
    *db = *nd;
    *nd = tmp;
    vecdb_free(nd);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */
static void emit_topk(const VecDB *db, Heap *res, int k, VecResult *out, int *n_out)
{
    int n = res->n < k ? res->n : k;
    while (res->n > n) heap_pop(res);
    for (int i = n - 1; i >= 0; i--) {
        HItem it = heap_pop(res);
        out[i].id = db->ids[it.node];
        out[i].dist = it.dist;
    }
    *n_out = n;
}

int vecdb_search_flat(const VecDB *db, const float *query, int k, VecResult *out)
{
    return vecdb_search_flat_batch(db, query, 1, k, out);
}

/* Blocked exact scan: QB queries share every vector load, with
 *   ||q - x||^2 = ||q||^2 - 2<q,x> + ||x||^2  (||x||^2 precomputed).
 * Arithmetic intensity rises QB-fold, lifting the scan off the
 * memory-bandwidth wall. */
#define FLAT_QB 8

#ifdef VECDB_AVX512
static void dots_block(const float *x, const float *const *qrows, int qb,
                       int dim, float *dots)
{
    __m512 acc[FLAT_QB];
    for (int j = 0; j < qb; j++) acc[j] = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        for (int j = 0; j < qb; j++)
            acc[j] = _mm512_fmadd_ps(xv, _mm512_loadu_ps(qrows[j] + i), acc[j]);
    }
    if (i < dim) {
        __mmask16 m = (__mmask16)((1u << (dim - i)) - 1u);
        __m512 xv = _mm512_maskz_loadu_ps(m, x + i);
        for (int j = 0; j < qb; j++)
            acc[j] = _mm512_fmadd_ps(xv, _mm512_maskz_loadu_ps(m, qrows[j] + i), acc[j]);
    }
    for (int j = 0; j < qb; j++) dots[j] = _mm512_reduce_add_ps(acc[j]);
}
#else
static void dots_block(const float *x, const float *const *qrows, int qb,
                       int dim, float *dots)
{
    for (int j = 0; j < qb; j++) {
        float s = 0.f;
        for (int i = 0; i < dim; i++) s += x[i] * qrows[j][i];
        dots[j] = s;
    }
}
#endif

int vecdb_search_flat_batch(const VecDB *db, const float *queries, int nq,
                            int k, VecResult *out)
{
    if (!db || db->alive == 0 || k <= 0 || nq <= 0) return 0;
    int kk = (size_t)k > db->alive ? (int)db->alive : k;

    for (int q0 = 0; q0 < nq; q0 += FLAT_QB) {
        int qb = nq - q0 < FLAT_QB ? nq - q0 : FLAT_QB;
        const float *qrows[FLAT_QB];
        float qn[FLAT_QB], dots[FLAT_QB];
        Heap res[FLAT_QB];
        for (int j = 0; j < qb; j++) {
            qrows[j] = queries + (size_t)(q0 + j) * db->dim;
            float s = 0.f;
            for (int i = 0; i < db->dim; i++) s += qrows[j][i] * qrows[j][i];
            qn[j] = s;
            heap_init(&res[j], kk + 1, 1);
        }

        for (size_t v = 0; v < db->count; v++) {
            if (db->dead[v]) continue;
            dots_block(vec_at(db, (uint32_t)v), qrows, qb, db->dim, dots);
            float xn = db->vnorms[v];
            for (int j = 0; j < qb; j++) {
                float d = qn[j] - 2.f * dots[j] + xn;
                if (d < 0.f) d = 0.f;            /* fp guard */
                if (res[j].n < kk) heap_push(&res[j], d, (uint32_t)v);
                else if (d < res[j].a[0].dist) {
                    heap_push(&res[j], d, (uint32_t)v);
                    heap_pop(&res[j]);
                }
            }
        }
        for (int j = 0; j < qb; j++) {
            int n;
            emit_topk(db, &res[j], kk, out + (size_t)(q0 + j) * k, &n);
            for (int t = n; t < k; t++) {        /* pad unused slots */
                out[(size_t)(q0 + j) * k + t].id = (uint64_t)-1;
                out[(size_t)(q0 + j) * k + t].dist = INFINITY;
            }
            heap_destroy(&res[j]);
        }
    }
    return kk;
}

int vecdb_search_hnsw(const VecDB *db_const, const float *query, int k, int ef, VecResult *out)
{
    VecDB *db = (VecDB *)db_const;   /* mutates visited/epoch only */
    if (!db || db->alive == 0 || k <= 0) return 0;
    if (ef < k) ef = k;

    uint32_t ep = (uint32_t)db->entry;
    for (int l = db->max_level; l > 0; l--)
        ep = greedy_step(db, query, ep, l);

    Heap res; heap_init(&res, ef + 1, 1);
    search_layer(db, query, ep, ef, 0, &res, 1);
    int n; emit_topk(db, &res, k, out, &n);
    heap_destroy(&res);
    return n;
}

/* ------------------------------------------------------------------ */
/* Persistence: [header][ids][vectors][per-node: level, links]         */
/* Format unchanged from v1 — files round-trip across layouts.         */
/* ------------------------------------------------------------------ */
#define VECDB_MAGIC 0x56454342u  /* "VECB" */

int vecdb_save(const VecDB *db, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = VECDB_MAGIC, version = 2;
    int64_t entry = db->entry;
    fwrite(&magic, 4, 1, f); fwrite(&version, 4, 1, f);
    fwrite(&db->dim, 4, 1, f); fwrite(&db->M, 4, 1, f);
    fwrite(&db->ef_construction, 4, 1, f); fwrite(&db->max_level, 4, 1, f);
    fwrite(&entry, 8, 1, f); fwrite(&db->count, 8, 1, f);
    fwrite(db->ids, sizeof(uint64_t), db->count, f);
    fwrite(db->vecs, sizeof(float), db->count * (size_t)db->dim, f);
    for (size_t i = 0; i < db->count; i++) {
        uint8_t lv = db->node_level[i];
        fwrite(&lv, 1, 1, f);
        for (int l = 0; l <= lv; l++) {
            const uint32_t *L = node_links(db, (uint32_t)i, l);
            uint32_t n = L[0];
            fwrite(&n, 4, 1, f);
            fwrite(&L[1], 4, n, f);
        }
    }
    fwrite(db->dead, 1, db->count, f);               /* v2: tombstones */
    fclose(f);
    return 0;
}

VecDB *vecdb_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
#define RD(ptr, sz, n) do { \
        if (fread((ptr), (sz), (n), f) != (size_t)(n)) goto fail; \
    } while (0)
    uint32_t magic = 0, version = 0;
    int dim, M, efc, max_level; int64_t entry; size_t count;
    VecDB *db = NULL;
    RD(&magic, 4, 1);
    if (magic != VECDB_MAGIC) goto fail;
    RD(&version, 4, 1);
    if (version < 1 || version > 2) goto fail;
    RD(&dim, 4, 1); RD(&M, 4, 1);
    RD(&efc, 4, 1); RD(&max_level, 4, 1);
    RD(&entry, 8, 1); RD(&count, 8, 1);

    VecDBConfig cfg = vecdb_default_config(dim);
    cfg.M = M; cfg.ef_construction = efc;
    cfg.initial_capacity = count ? count : 1024;
    db = vecdb_create(&cfg);
    db->count = count; db->entry = entry; db->max_level = max_level;

    RD(db->ids, sizeof(uint64_t), count);
    RD(db->vecs, sizeof(float), count * (size_t)dim);
    for (size_t i = 0; i < count; i++) {
        double nsq = 0.0;
        const float *v = db->vecs + i * (size_t)dim;
        for (int j = 0; j < dim; j++) nsq += (double)v[j] * v[j];
        db->vnorms[i] = (float)nsq;
    }
    for (size_t i = 0; i < count; i++) {
        uint8_t lv;
        RD(&lv, 1, 1);
        db->node_level[i] = lv;
        if (lv > 0)
            db->upper[i] = calloc((size_t)lv * (size_t)(db->M + 2), sizeof(uint32_t));
        for (int l = 0; l <= lv; l++) {
            uint32_t n = 0;
            RD(&n, 4, 1);
            if (n > (uint32_t)link_cap(db, l) + 1) goto fail;
            uint32_t *L = node_links(db, (uint32_t)i, l);
            L[0] = n;
            RD(&L[1], 4, n);
            for (uint32_t j = 1; j <= n; j++)        /* validate link range */
                if (L[j] >= count) goto fail;
        }
    }
    if (version >= 2) RD(db->dead, 1, count);
    db->alive = 0;
    for (size_t i = 0; i < count; i++) {
        if (db->dead[i]) continue;
        db->alive++;
        map_put(db, db->ids[i], (uint32_t)i);
    }
    fclose(f);
    return db;
fail:
    fclose(f);
    vecdb_free(db);
    return NULL;
#undef RD
}
