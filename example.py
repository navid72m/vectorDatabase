import numpy as np
from pyvecdb import VecDB, TQIndex


def build_and_query_vecdb():
    dim = 128
    n = 1000

    print("Building VecDB index...")
    db = VecDB(dim=dim, M=16, ef_construction=200)

    vecs = np.random.rand(n, dim).astype(np.float32)
    db.add(vecs)

    queries = vecs[:5]
    print(f"Searching {len(queries)} queries with k=10, ef=100...")
    ids, dists = db.search(queries, k=10, ef=100)

    print("HNSW search results:")
    for qi, (row_ids, row_dists) in enumerate(zip(ids, dists)):
        print(f"query {qi}: ids={row_ids[:5].tolist()} dists={row_dists[:5].tolist()}")

    db.save("index.vecdb")
    print("Saved index.vecdb")

    db2 = VecDB.load("index.vecdb")
    ids2, dists2 = db2.search(queries, k=5, ef=100)
    print("Loaded index search example:")
    print(ids2[:1], dists2[:1])


def build_and_query_tqindex():
    dim = 128
    n = 1000

    print("Building TurboQuant index...")
    tq = TQIndex(dim=dim, bits=4, qjl=False)

    vecs = np.random.rand(n, dim).astype(np.float32)
    tq.add(vecs)

    queries = vecs[:5]
    print(f"Searching {len(queries)} queries with k=10...")
    ids, dists = tq.search(queries, k=10)

    print("TurboQuant search results:")
    for qi, (row_ids, row_dists) in enumerate(zip(ids, dists)):
        print(f"query {qi}: ids={row_ids[:5].tolist()} dists={row_dists[:5].tolist()}")


if __name__ == "__main__":
    print("Running Vector Database example")
    build_and_query_vecdb()
    print("\nRunning TurboQuant example")
    build_and_query_tqindex()
