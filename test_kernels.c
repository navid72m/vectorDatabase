/* test_kernels.c — architecture-portable correctness harness.
 * Runs the SIMD kernels (AVX-512, NEON, or scalar — whatever the build
 * selected) against independent scalar references. Used under QEMU to
 * validate the aarch64/NEON paths without ARM hardware.
 * Exit 0 = pass.
 */
#include "vecdb.h"
#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static uint64_t rs = 0x123456789ULL;
static float frnd(void) {
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return (float)((rs * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1 << 24) - 0.5f;
}

static int FAILS = 0;
static void check(const char *name, int cond) {
    printf("  [%s] %s\n", cond ? "ok" : "FAIL", name);
    if (!cond) FAILS++;
}

int main(void) {
    const int N = 1500, DIM = 128, Q = 30, K = 5;
    float *data = malloc((size_t)N * DIM * sizeof(float));
    for (int i = 0; i < N * DIM; i++) data[i] = frnd();

    /* ---- exact search vs scalar reference ---- */
    VecDBConfig cfg = vecdb_default_config(DIM);
    VecDB *db = vecdb_create(&cfg);
    for (int i = 0; i < N; i++) vecdb_add(db, (uint64_t)i, data + (size_t)i * DIM);

    int agree = 1; float maxerr = 0.f;
    for (int q = 0; q < Q; q++) {
        const float *qv = data + (size_t)q * DIM;
        VecResult out[8];
        vecdb_search_flat(db, qv, K, out);
        /* scalar reference top-1 + distance check on every result */
        for (int j = 0; j < K; j++) {
            const float *x = data + (size_t)out[j].id * DIM;
            double d = 0;
            for (int t = 0; t < DIM; t++) { double e = qv[t] - x[t]; d += e * e; }
            float err = fabsf((float)d - out[j].dist);
            if (err > maxerr) maxerr = err;
        }
        if (out[0].id != (uint64_t)q || out[0].dist > 1e-3f) agree = 0;
    }
    check("flat self-NN exact", agree);
    check("flat distances match scalar (<1e-2)", maxerr < 1e-2f);

    /* ---- HNSW self-NN ---- */
    int hits = 0;
    for (int q = 0; q < Q; q++) {
        VecResult out[8];
        vecdb_search_hnsw(db, data + (size_t)q * DIM, 1, 50, out);
        hits += (out[0].id == (uint64_t)q);
    }
    check("hnsw self-NN >= 28/30", hits >= 28);

    /* ---- delete + filter ---- */
    for (int i = 0; i < 300; i++) vecdb_delete(db, (uint64_t)i);
    int leaked = 0;
    for (int q = 0; q < Q; q++) {
        VecResult out[8];
        int n = vecdb_search_hnsw(db, data + (size_t)(q + 500) * DIM, K, 100, out);
        for (int j = 0; j < n; j++) if (out[j].id < 300) leaked++;
    }
    check("tombstones: no deleted ids returned", leaked == 0);

    /* ---- TurboQuant 4/8-bit self-NN (exercises tbl/sdot or permute/dpbusd) */
    for (int bits = 4; bits <= 8; bits += 4) {
        TQIndex *tq = tq_create(DIM, bits, 7);
        for (int i = 0; i < N; i++) tq_add(tq, (uint64_t)i, data + (size_t)i * DIM);
        int ok = 0;
        for (int q = 0; q < Q; q++) {
            VecResult out[8];
            tq_search(tq, data + (size_t)q * DIM, 1, out);
            ok += (out[0].id == (uint64_t)q);
        }
        char name[64];
        snprintf(name, sizeof name, "turboquant %d-bit self-NN >= 28/30 (%d)", bits, ok);
        check(name, ok >= 28);
        tq_free(tq);
    }

    vecdb_free(db);
    free(data);
    printf("%s\n", FAILS ? "KERNEL TESTS FAILED" : "ALL KERNEL TESTS PASS");
    return FAILS ? 1 : 0;
}
