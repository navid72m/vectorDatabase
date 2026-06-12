/* turboquant.c — see turboquant.h */
#include "turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* Fast Walsh-Hadamard transform, in place, n must be a power of two.  */
/* ------------------------------------------------------------------ */
static void fht(float *a, int n)
{
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float u = a[j], v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
}

static int next_pow2(int x) { int p = 1; while (p < x) p <<= 1; return p; }

/* xorshift64* */
static uint64_t rng_next(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* ------------------------------------------------------------------ */
/* Lloyd-Max quantizer for N(0,1): codebook levels + decision bounds.  */
/* Centroid of an interval under the Gaussian has closed form:         */
/*   E[X | a<X<b] = (pdf(a) - pdf(b)) / (cdf(b) - cdf(a))              */
/* so the fixed-point iteration is exact — no sampling, no data.       */
/* ------------------------------------------------------------------ */
static double npdf(double x) { return exp(-0.5 * x * x) / sqrt(2.0 * M_PI); }
static double ncdf(double x) { return 0.5 * (1.0 + erf(x / sqrt(2.0))); }

static void lloyd_max_gaussian(int K, float *levels, float *bounds /* K-1 */)
{
    double *L = malloc(K * sizeof(double));
    double *B = malloc((K + 1) * sizeof(double));
    for (int k = 0; k < K; k++) L[k] = -3.0 + 6.0 * (k + 0.5) / K;

    for (int it = 0; it < 300; it++) {
        B[0] = -1e30; B[K] = 1e30;
        for (int k = 1; k < K; k++) B[k] = 0.5 * (L[k-1] + L[k]);
        for (int k = 0; k < K; k++) {
            double pa = (B[k]   <= -1e29) ? 0.0 : npdf(B[k]);
            double pb = (B[k+1] >=  1e29) ? 0.0 : npdf(B[k+1]);
            double ca = (B[k]   <= -1e29) ? 0.0 : ncdf(B[k]);
            double cb = (B[k+1] >=  1e29) ? 1.0 : ncdf(B[k+1]);
            double mass = cb - ca;
            if (mass > 1e-300) L[k] = (pa - pb) / mass;
        }
    }
    for (int k = 0; k < K; k++) levels[k] = (float)L[k];
    for (int k = 0; k < K - 1; k++) bounds[k] = (float)(0.5 * (L[k] + L[k+1]));
    free(L); free(B);
}

/* ------------------------------------------------------------------ */
#define TQ_ROUNDS 3

struct TQIndex {
    int dim;          /* user-facing dimensionality            */
    int pdim;         /* padded to power of two                */
    int bits;         /* 4 or 8                                */
    int K;            /* 1 << bits                             */
    size_t code_bytes;/* per-vector packed code size           */

    float *levels;    /* K Lloyd-Max levels for N(0,1)         */
    float *bounds;    /* K-1 decision boundaries               */
    float *signs;     /* TQ_ROUNDS * pdim, each +-1            */
    float  fht_scale; /* (1/sqrt(pdim))^TQ_ROUNDS              */

    uint8_t  *codes;  /* count * code_bytes                    */
    float    *norms;  /* count                                 */
    float    *csqs;   /* count: sum of levels[c_i]^2 (unit scale) */

    /* bits==8 stores re-quantized int8 values (Lloyd-Max reconstruction
     * snapped to a signed-127 grid) so the scan can use AVX512-VNNI.    */
    float     lmax;   /* max |level|                           */
    float     s8x;    /* 127 / lmax: float -> int8 scale       */
    int8_t    l8[16]; /* bits==4: codebook on the int8 grid (NEON tbl) */
    int32_t  *i8sums; /* count: sum of stored int8 (VNNI bias) */

    /* QJL residual stage (TurboQuant-prod, Alg. 2 of the paper)   */
    int       qjl;        /* 0 = MSE variant only                  */
    float    *S;          /* pdim x pdim iid N(0,1) projection     */
    uint8_t  *qjl_bits;   /* count * qjl_bytes packed sign bits    */
    float    *rnorms;     /* count: ||residual||_2                 */
    size_t    qjl_bytes;  /* pdim/8 per vector                     */
    float    *scratch2;   /* pdim, for S*r matvec                  */
    uint64_t *ids;    /* count                                 */
    size_t count, cap;

    float *scratch;   /* pdim, reused for rotation             */
};

/* Box-Muller standard normal from xorshift */
static double rng_gauss(uint64_t *s)
{
    double u1 = ((double)(rng_next(s) >> 11) + 1.0) / 9007199254740994.0;
    double u2 = ((double)(rng_next(s) >> 11) + 1.0) / 9007199254740994.0;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

TQIndex *tq_create2(int dim, int bits, int use_qjl, uint64_t seed)
{
    if (dim <= 0 || (bits != 4 && bits != 8)) return NULL;
    TQIndex *tq = calloc(1, sizeof(TQIndex));
    tq->dim = dim;
    tq->pdim = next_pow2(dim);
    tq->bits = bits;
    tq->K = 1 << bits;
    tq->code_bytes = ((size_t)tq->pdim * bits + 7) / 8;

    tq->levels = malloc(tq->K * sizeof(float));
    tq->bounds = malloc((tq->K - 1) * sizeof(float));
    lloyd_max_gaussian(tq->K, tq->levels, tq->bounds);
    tq->lmax = 0.f;
    for (int k = 0; k < tq->K; k++) {
        float a = fabsf(tq->levels[k]);
        if (a > tq->lmax) tq->lmax = a;
    }
    tq->s8x = 127.0f / tq->lmax;
    if (tq->K <= 16)
        for (int k = 0; k < tq->K; k++) {
            int v = (int)lrintf(tq->levels[k] * tq->s8x);
            tq->l8[k] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
        }

    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    tq->signs = malloc((size_t)TQ_ROUNDS * tq->pdim * sizeof(float));
    for (size_t i = 0; i < (size_t)TQ_ROUNDS * tq->pdim; i++)
        tq->signs[i] = (rng_next(&s) >> 63) ? 1.0f : -1.0f;
    tq->fht_scale = (float)pow(1.0 / sqrt((double)tq->pdim), TQ_ROUNDS);

    tq->qjl = use_qjl ? 1 : 0;
    if (tq->qjl) {
        tq->qjl_bytes = (size_t)tq->pdim / 8;
        tq->S = malloc((size_t)tq->pdim * tq->pdim * sizeof(float));
        for (size_t i = 0; i < (size_t)tq->pdim * tq->pdim; i++)
            tq->S[i] = (float)rng_gauss(&s);
        tq->scratch2 = malloc(tq->pdim * sizeof(float));
    }

    tq->cap = 1024;
    tq->codes = malloc(tq->cap * tq->code_bytes);
    tq->i8sums = bits == 8 ? malloc(tq->cap * sizeof(int32_t)) : NULL;
    tq->norms = malloc(tq->cap * sizeof(float));
    tq->csqs  = malloc(tq->cap * sizeof(float));
    tq->ids   = malloc(tq->cap * sizeof(uint64_t));
    tq->scratch = malloc(tq->pdim * sizeof(float));
    if (tq->qjl) {
        tq->qjl_bits = malloc(tq->cap * tq->qjl_bytes);
        tq->rnorms   = malloc(tq->cap * sizeof(float));
    }
    return tq;
}

TQIndex *tq_create(int dim, int bits, uint64_t seed)
{
    return tq_create2(dim, bits, 0, seed);
}

void tq_free(TQIndex *tq)
{
    if (!tq) return;
    free(tq->levels); free(tq->bounds); free(tq->signs);
    free(tq->codes); free(tq->i8sums); free(tq->norms); free(tq->csqs); free(tq->ids); free(tq->scratch);
    free(tq->S); free(tq->qjl_bits); free(tq->rnorms); free(tq->scratch2);
    free(tq);
}

size_t tq_count(const TQIndex *tq) { return tq->count; }

size_t tq_memory_bytes(const TQIndex *tq)
{
    size_t per = tq->code_bytes + 2 * sizeof(float) + sizeof(uint64_t);
    if (tq->bits == 8) per += sizeof(int32_t);          /* VNNI bias sum */
    if (tq->qjl) per += tq->qjl_bytes + sizeof(float);
    return tq->count * per;
}

/* Rotate (in place, length pdim): 3 rounds of sign-flip + Hadamard,
 * scaled to keep the transform orthonormal.                            */
static void tq_rotate(const TQIndex *tq, float *x)
{
    for (int r = 0; r < TQ_ROUNDS; r++) {
        const float *sg = tq->signs + (size_t)r * tq->pdim;
        for (int i = 0; i < tq->pdim; i++) x[i] *= sg[i];
        fht(x, tq->pdim);
    }
    for (int i = 0; i < tq->pdim; i++) x[i] *= tq->fht_scale;
}

/* Nearest codeword via binary search over decision boundaries. */
static inline int quantize_coord(const TQIndex *tq, float v)
{
    int lo = 0, hi = tq->K - 1;          /* invariant: answer in [lo, hi] */
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (v <= tq->bounds[mid]) hi = mid; else lo = mid + 1;
    }
    return lo;
}

static float tq_dotf(const float *a, const float *b, int n);

int64_t tq_add(TQIndex *tq, uint64_t id, const float *vec)
{
    if (!tq || !vec) return -1;
    if (tq->count == tq->cap) {
        tq->cap *= 2;
        tq->codes = realloc(tq->codes, tq->cap * tq->code_bytes);
        if (tq->i8sums) tq->i8sums = realloc(tq->i8sums, tq->cap * sizeof(int32_t));
        tq->norms = realloc(tq->norms, tq->cap * sizeof(float));
        tq->csqs  = realloc(tq->csqs,  tq->cap * sizeof(float));
        tq->ids   = realloc(tq->ids,   tq->cap * sizeof(uint64_t));
        if (tq->qjl) {
            tq->qjl_bits = realloc(tq->qjl_bits, tq->cap * tq->qjl_bytes);
            tq->rnorms   = realloc(tq->rnorms,   tq->cap * sizeof(float));
        }
    }

    float *x = tq->scratch;
    memcpy(x, vec, (size_t)tq->dim * sizeof(float));
    memset(x + tq->dim, 0, (size_t)(tq->pdim - tq->dim) * sizeof(float));

    /* norm separation: quantize the direction only */
    double nsq = 0.0;
    for (int i = 0; i < tq->dim; i++) nsq += (double)x[i] * x[i];
    float norm = (float)sqrt(nsq);
    if (norm > 0.f) {
        float inv = 1.0f / norm;
        for (int i = 0; i < tq->dim; i++) x[i] *= inv;
    }

    /* rotate; unit vector coords are ~N(0, 1/pdim), so scale by sqrt(pdim)
     * to match the N(0,1) codebook */
    tq_rotate(tq, x);
    float sd = sqrtf((float)tq->pdim);

    uint8_t *code = tq->codes + tq->count * tq->code_bytes;
    memset(code, 0, tq->code_bytes);
    double csq = 0.0;
    int32_t i8sum = 0;
    for (int i = 0; i < tq->pdim; i++) {
        int c = quantize_coord(tq, x[i] * sd);
        if (tq->bits == 8) {
            /* snap the Lloyd-Max reconstruction to the int8 grid; csq and
             * the QJL residual use the SAME stored value, so the distance
             * decomposition stays exact w.r.t. what the scan computes.   */
            int q8 = (int)lrintf(tq->levels[c] * tq->s8x);
            if (q8 >  127) q8 =  127;
            if (q8 < -127) q8 = -127;
            ((int8_t *)code)[i] = (int8_t)q8;
            float v = (float)q8 / tq->s8x;
            csq += (double)v * v;
            i8sum += q8;
        } else {
            code[i >> 1] |= (uint8_t)(c << ((i & 1) ? 4 : 0));
            csq += (double)tq->levels[c] * tq->levels[c];
        }
    }
    if (tq->bits == 8) tq->i8sums[tq->count] = i8sum;
    tq->norms[tq->count] = norm;
    tq->csqs[tq->count] = (float)csq;

    if (tq->qjl) {
        /* residual in rotated unit space: r_i = y_i - L[c_i]/sqrt(pdim).
         * x[] still holds the rotated unit vector here. */
        float inv_sdf = 1.0f / sd;
        float *r = tq->scratch2;
        double rn = 0.0;
        for (int i = 0; i < tq->pdim; i++) {
            float recon;
            if (tq->bits == 8) recon = (float)((int8_t *)code)[i] / tq->s8x;
            else recon = tq->levels[(code[i >> 1] >> ((i & 1) ? 4 : 0)) & 0x0F];
            r[i] = x[i] - recon * inv_sdf;
            rn += (double)r[i] * r[i];
        }
        tq->rnorms[tq->count] = (float)sqrt(rn);

        uint8_t *bits_out = tq->qjl_bits + tq->count * tq->qjl_bytes;
        memset(bits_out, 0, tq->qjl_bytes);
        for (int i = 0; i < tq->pdim; i++) {        /* sign(S * r) */
            float dotp = tq_dotf(tq->S + (size_t)i * tq->pdim, r, tq->pdim);
            if (dotp >= 0.f) bits_out[i >> 3] |= (uint8_t)(1u << (i & 7));
        }
    }

    tq->ids[tq->count] = id;
    return (int64_t)tq->count++;
}

/* ------------------------------------------------------------------ */
/* Asymmetric search via the decomposition                             */
/*   ||q - x_hat||^2 = ||Rq||^2 - 2*s*<Rq, L[c]> + s^2 * ||L[c]||^2    */
/* with s = norm/sqrt(pdim); ||L[c]||^2 precomputed at encode time.    */
/* The scan reduces to a dot product between the rotated query and     */
/* decoded codes — SIMD: 4-bit decodes via a register-resident LUT     */
/* (_mm512_permutexvar_ps, the whole codebook lives in one zmm),       */
/* 8-bit stores re-quantized int8 and scans with AVX512-VNNI dpbusd.   */
/* ------------------------------------------------------------------ */
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

static float tq_dot4_avx512(const float *qr, const uint8_t *code, int pdim,
                            const float *levels)
{
    const __m512 lv = _mm512_loadu_ps(levels);           /* 16 codewords */
    const __m128i nib = _mm_set1_epi8(0x0F);
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < pdim; i += 16) {
        __m128i raw = _mm_loadl_epi64((const __m128i *)(code + (i >> 1)));
        __m128i lo  = _mm_and_si128(raw, nib);                    /* even coords */
        __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), nib); /* odd coords  */
        __m512i idx = _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi));
        __m512 vals = _mm512_permutexvar_ps(idx, lv);
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(qr + i), vals, acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#define TQ_HAVE_SIMD 1
#endif

#if defined(__AVX512VNNI__)
/* int8 dot via vpdpbusd: 64 MACs/instruction. Query is offset to unsigned
 * (qu8 = qi8 + 128); caller corrects with  - 128 * sum(x_i8).             */
static int32_t tq_doti8_vnni(const uint8_t *qu8, const int8_t *x, int pdim)
{
    __m512i acc = _mm512_setzero_si512();
    for (int i = 0; i < pdim; i += 64)
        acc = _mm512_dpbusd_epi32(acc,
                _mm512_loadu_si512((const void *)(qu8 + i)),
                _mm512_loadu_si512((const void *)(x + i)));
    return _mm512_reduce_add_epi32(acc);
}
#define TQ_HAVE_VNNI 1
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
/* Both bit-widths scan as signed-int8 dots via SDOT (4 MACs/lane,
 * 16 codes per instruction). No zero-point correction needed: SDOT is
 * signed x signed, unlike AVX512-VNNI's unsigned x signed dpbusd.
 * 4-bit decode: nibbles -> vqtbl1q over the 16-entry int8 codebook,
 * which fits exactly in one table register.                            */
static int32_t tq_dot4_neon(const int8_t *qi8, const uint8_t *code, int pdim,
                            const int8_t *l8)
{
    const int8x16_t tbl = vld1q_s8(l8);
    const uint8x8_t nib = vdup_n_u8(0x0F);
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < pdim; i += 16) {
        uint8x8_t raw = vld1_u8(code + (i >> 1));        /* 8 bytes = 16 codes */
        uint8x8_t lo = vand_u8(raw, nib);                /* even coords */
        uint8x8_t hi = vshr_n_u8(raw, 4);                /* odd coords  */
        uint8x8x2_t z = vzip_u8(lo, hi);                 /* interleave  */
        uint8x16_t idx = vcombine_u8(z.val[0], z.val[1]);
        int8x16_t vals = vqtbl1q_s8(tbl, idx);
        acc = vdotq_s32(acc, vals, vld1q_s8(qi8 + i));
    }
    return vaddvq_s32(acc);
}

static int32_t tq_dot8_neon(const int8_t *qi8, const int8_t *x, int pdim)
{
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int i = 0;
    for (; i + 32 <= pdim; i += 32) {
        a0 = vdotq_s32(a0, vld1q_s8(x + i),      vld1q_s8(qi8 + i));
        a1 = vdotq_s32(a1, vld1q_s8(x + i + 16), vld1q_s8(qi8 + i + 16));
    }
    for (; i + 16 <= pdim; i += 16)
        a0 = vdotq_s32(a0, vld1q_s8(x + i), vld1q_s8(qi8 + i));
    return vaddvq_s32(vaddq_s32(a0, a1));
}
#define TQ_NEON_DOT 1
#endif

#if !defined(TQ_NEON_DOT)
static int32_t tq_doti8_scalar(const uint8_t *qu8, const int8_t *x, int pdim)
{
    int32_t s = 0;
    for (int i = 0; i < pdim; i++) s += (int32_t)qu8[i] * (int32_t)x[i];
    return s;
}
#endif

/* plain float dot with AVX-512 + masked tail (QJL matvecs) */
#ifdef TQ_HAVE_SIMD
static float tq_dotf(const float *a, const float *b, int n)
{
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    if (i < n) {
        __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
        s += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(m, a + i),
                                                _mm512_maskz_loadu_ps(m, b + i)));
    }
    return s;
}
#else
static float tq_dotf(const float *a, const float *b, int n)
{
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

static float tq_dot_scalar(const TQIndex *tq, const float *qr, const uint8_t *code)
{
    float dot = 0.f;
    if (tq->bits == 8) {
        const int8_t *x = (const int8_t *)code;
        for (int i = 0; i < tq->pdim; i++) dot += qr[i] * (float)x[i];
        dot /= tq->s8x;
    } else {
        for (int i = 0; i < tq->pdim; i += 2) {
            uint8_t b = code[i >> 1];
            dot += qr[i] * tq->levels[b & 0x0F] + qr[i+1] * tq->levels[b >> 4];
        }
    }
    return dot;
}

/* Sign-dot for QJL: sum of Sq[i] with the sign given by bit i.
 * AVX-512: stored bits become a __mmask16 directly — negate-where-clear. */
#ifdef TQ_HAVE_SIMD
static float tq_signdot_avx512(const float *Sq, const uint8_t *bits, int pdim)
{
    __m512 acc = _mm512_setzero_ps();
    const __m512 zero = _mm512_setzero_ps();
    for (int i = 0; i < pdim; i += 16) {
        __mmask16 neg = (__mmask16)~(((const uint16_t *)bits)[i >> 4]);
        __m512 v = _mm512_loadu_ps(Sq + i);
        acc = _mm512_add_ps(acc, _mm512_mask_sub_ps(v, neg, zero, v));
    }
    return _mm512_reduce_add_ps(acc);
}
#endif

static float tq_signdot_scalar(const float *Sq, const uint8_t *bits, int pdim)
{
    float s = 0.f;
    for (int i = 0; i < pdim; i++)
        s += (bits[i >> 3] >> (i & 7)) & 1 ? Sq[i] : -Sq[i];
    return s;
}

int tq_search(const TQIndex *tq_c, const float *query, int k, VecResult *out)
{
    TQIndex *tq = (TQIndex *)tq_c;   /* uses scratch only */
    if (!tq || tq->count == 0 || k <= 0) return 0;
    if ((size_t)k > tq->count) k = (int)tq->count;

    float *qr = malloc(tq->pdim * sizeof(float));
    memcpy(qr, query, (size_t)tq->dim * sizeof(float));
    memset(qr + tq->dim, 0, (size_t)(tq->pdim - tq->dim) * sizeof(float));
    tq_rotate(tq, qr);

    float qn = 0.f;
    for (int i = 0; i < tq->pdim; i++) qn += qr[i] * qr[i];

    /* QJL query prep: Sq = S * qr (once per query), plus the estimator's
     * sqrt(pi/2)/pdim constant from Definition 1 of the paper. */
    float *Sq = NULL;
    float qjl_const = 0.f;
    if (tq->qjl) {
        Sq = malloc(tq->pdim * sizeof(float));
        for (int i = 0; i < tq->pdim; i++)
            Sq[i] = tq_dotf(tq->S + (size_t)i * tq->pdim, qr, tq->pdim);
        qjl_const = sqrtf((float)M_PI / 2.0f) / (float)tq->pdim;
    }

#ifdef TQ_HAVE_SIMD
    const int simd = (tq->pdim % 16) == 0;
#endif

    /* int8 query prep: quantize qr once per query. NEON+SDOT scans both
     * bit-widths as signed int8 (4-bit decodes via tbl); x86 keeps the
     * float permutexvar path for 4-bit and uses VNNI (unsigned+bias) or
     * scalar for 8-bit. */
    uint8_t *qu8 = NULL;          /* unsigned + 128 offset (VNNI/scalar) */
    int8_t  *qi8 = NULL;          /* signed (NEON SDOT)                  */
    float i8_inv_scale = 0.f;     /* 1 / (s8q * s8x) */
    int use_i8 = 0;
#if defined(TQ_NEON_DOT)
    int want_i8 = (tq->pdim % 16) == 0;            /* 4-bit and 8-bit */
#else
    int want_i8 = (tq->bits == 8);
#endif
    if (want_i8) {
        float qmax = 0.f;
        for (int i = 0; i < tq->pdim; i++) {
            float a = fabsf(qr[i]);
            if (a > qmax) qmax = a;
        }
        if (qmax > 0.f) {
            float s8q = 127.0f / qmax;
            qi8 = malloc(tq->pdim);
            for (int i = 0; i < tq->pdim; i++) {
                int q = (int)lrintf(qr[i] * s8q);
                if (q >  127) q =  127;
                if (q < -127) q = -127;
                qi8[i] = (int8_t)q;
            }
#if !defined(TQ_NEON_DOT)
            qu8 = malloc(tq->pdim);
            for (int i = 0; i < tq->pdim; i++)
                qu8[i] = (uint8_t)((int)qi8[i] + 128);
#endif
            i8_inv_scale = 1.0f / (s8q * tq->s8x);
            use_i8 = 1;
        }
    }

    /* simple bounded max-heap over (dist, idx) */
    float    *hd = malloc((size_t)k * sizeof(float));
    uint32_t *hi = malloc((size_t)k * sizeof(uint32_t));
    int hn = 0;
    float inv_sd = 1.0f / sqrtf((float)tq->pdim);

    for (size_t v = 0; v < tq->count; v++) {
        const uint8_t *code = tq->codes + v * tq->code_bytes;
        float dot;
        if (use_i8) {
#if defined(TQ_NEON_DOT)
            int32_t di = (tq->bits == 4)
                ? tq_dot4_neon(qi8, code, tq->pdim, tq->l8)
                : tq_dot8_neon(qi8, (const int8_t *)code, tq->pdim);
            dot = (float)di * i8_inv_scale;     /* signed x signed: no bias */
#else
            int32_t di;
#ifdef TQ_HAVE_VNNI
            if ((tq->pdim % 64) == 0)
                di = tq_doti8_vnni(qu8, (const int8_t *)code, tq->pdim);
            else
#endif
                di = tq_doti8_scalar(qu8, (const int8_t *)code, tq->pdim);
            dot = (float)(di - 128 * tq->i8sums[v]) * i8_inv_scale;
#endif
        }
#ifdef TQ_HAVE_SIMD
        else if (simd && tq->bits == 4)
            dot = tq_dot4_avx512(qr, code, tq->pdim, tq->levels);
#endif
        else
            dot = tq_dot_scalar(tq, qr, code);

        float s = tq->norms[v] * inv_sd;
        float d;
        if (tq->qjl) {
            /* unbiased <q,x> = n*(<qr,y_hat> + ||r||*qjl_est), then
             * ||q-x||^2 = ||q||^2 - 2<q,x> + ||x||^2 with the TRUE norm */
            const uint8_t *qb = tq->qjl_bits + v * tq->qjl_bytes;
            float sd_dot;
#ifdef TQ_HAVE_SIMD
            if (simd) sd_dot = tq_signdot_avx512(Sq, qb, tq->pdim);
            else
#endif
                sd_dot = tq_signdot_scalar(Sq, qb, tq->pdim);
            float ip_unit = dot * inv_sd + tq->rnorms[v] * qjl_const * sd_dot;
            float n = tq->norms[v];
            d = qn - 2.f * n * ip_unit + n * n;
        } else {
            d = qn - 2.f * s * dot + s * s * tq->csqs[v];
        }

        if (hn < k) {                       /* push */
            int j = hn++;
            hd[j] = d; hi[j] = (uint32_t)v;
            while (j > 0) {
                int p = (j - 1) >> 1;
                if (hd[p] >= hd[j]) break;
                float td = hd[p]; hd[p] = hd[j]; hd[j] = td;
                uint32_t ti = hi[p]; hi[p] = hi[j]; hi[j] = ti;
                j = p;
            }
        } else if (d < hd[0]) {             /* replace max, sift down */
            hd[0] = d; hi[0] = (uint32_t)v;
            int j = 0;
            for (;;) {
                int l = 2*j+1, r = 2*j+2, m = j;
                if (l < k && hd[l] > hd[m]) m = l;
                if (r < k && hd[r] > hd[m]) m = r;
                if (m == j) break;
                float td = hd[m]; hd[m] = hd[j]; hd[j] = td;
                uint32_t ti = hi[m]; hi[m] = hi[j]; hi[j] = ti;
                j = m;
            }
        }
    }

    /* pop into ascending order */
    for (int n = hn; n > 0; n--) {
        out[n-1].id = tq->ids[hi[0]];
        out[n-1].dist = hd[0];
        hd[0] = hd[n-1]; hi[0] = hi[n-1];
        int j = 0, lim = n - 1;
        for (;;) {
            int l = 2*j+1, r = 2*j+2, m = j;
            if (l < lim && hd[l] > hd[m]) m = l;
            if (r < lim && hd[r] > hd[m]) m = r;
            if (m == j) break;
            float td = hd[m]; hd[m] = hd[j]; hd[j] = td;
            uint32_t ti = hi[m]; hi[m] = hi[j]; hi[j] = ti;
            j = m;
        }
    }
    free(qr); free(Sq); free(qu8); free(qi8); free(hd); free(hi);
    return hn;
}

/* Batched scan, parallel over queries when built with OpenMP.
 * tq_search allocates all per-query scratch locally, so concurrent
 * searches share no mutable state. */
int tq_search_batch(const TQIndex *tq, const float *queries, int nq, int k,
                    VecResult *out)
{
    if (!tq || nq <= 0 || k <= 0) return 0;
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < nq; i++) {
        VecResult *o = out + (size_t)i * k;
        int n = tq_search(tq, queries + (size_t)i * tq->dim, k, o);
        for (int j = n; j < k; j++) { o[j].id = (uint64_t)-1; o[j].dist = INFINITY; }
    }
    return k;
}
