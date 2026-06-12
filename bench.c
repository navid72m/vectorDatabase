#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vecdb.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_vectors> <dim>\n", argv[0]);
        return 1;
    }
    size_t n = (size_t)atoi(argv[1]);
    size_t dim = (size_t)atoi(argv[2]);
    VecDBConfig cfg = vecdb_default_config(dim);
    VecDB *db = vecdb_create(&cfg);

    // generate random vectors
    srand((unsigned)time(NULL));
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
    vecdb_search_hnsw(db, q, 10, 100, out);
    printf("Top result id: %llu distance: %f\n", (unsigned long long)out[0].id, out[0].dist);
    free(q);
    vecdb_save(db, "index.vecdb");
    vecdb_free(db);
    return 0;
}
