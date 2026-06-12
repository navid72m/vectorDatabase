#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vecdb.h"

int main(int argc, char **argv) {
    // New usage: bench <num_vectors> <dim> [reps]
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <num_vectors> <dim> [reps]\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atoi(argv[1]);
    size_t dim = (size_t)atoi(argv[2]);
    int reps = (argc == 4) ? atoi(argv[3]) : 1;
    if (reps < 1) reps = 1;
    struct timespec ts_start, ts_end;
    double total_sec = 0.0;

    for (int r = 0; r < reps; ++r) {
        VecDBConfig cfg = vecdb_default_config(dim);
        VecDB *db = vecdb_create(&cfg);

        // generate random vectors
        srand((unsigned)time(NULL) + r); // vary seed per rep
        for (size_t i = 0; i < n; i++) {
            float *vec = (float *)malloc(dim * sizeof(float));
            for (size_t d = 0; d < dim; d++) {
                vec[d] = (float)rand() / RAND_MAX;
            }
            vecdb_add(db, (uint64_t)i, vec);
            free(vec);
        }

        // simple query
        float *q = (float *)malloc(dim * sizeof(float));
        for (size_t d = 0; d < dim; d++) {
            q[d] = (float)rand() / RAND_MAX;
        }
        VecResult out[10];
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        vecdb_search_hnsw(db, q, 10, 100, out);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double sec = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
        total_sec += sec;
        printf("Run %d: top id %llu dist %f, query time %.6f s\n", r+1, (unsigned long long)out[0].id, out[0].dist, sec);
        free(q);
        vecdb_save(db, "index.vecdb");
        vecdb_free(db);
    }

    double avg = total_sec / reps;
    printf("Average query time over %d runs: %.6f s\n", reps, avg);
    return 0;
}
