"""vecdb test suite — run with: python tests.py

Covers: exact/HNSW correctness, persistence round-trips, tombstone deletes
(no leakage, no duplicates, stable recall under churn), compact, and edge
cases. Exits non-zero on any failure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pyvecdb import VecDB, TQIndex

FAILS = 0
def _raises(fn):
    try: fn(); return False
    except Exception: return True

def check(name, cond, detail=""):
    global FAILS
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not cond: FAILS += 1

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- basics
print("basics:")
X = rng.standard_normal((3000, 64)).astype(np.float32)
db = VecDB(dim=64); db.add(X)
Q = rng.standard_normal((50, 64)).astype(np.float32)
D = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
gt = np.argsort(D, axis=1)[:, :10]
e_ids, e_d = db.search(Q, k=10, exact=True)
check("exact matches numpy", np.array_equal(e_ids.astype(np.int64), gt))
h_ids, _ = db.search(Q, k=10, ef=200)
rec = np.mean([len(set(a.tolist()) & set(b.tolist()))/10
               for a, b in zip(h_ids.astype(np.int64), gt)])
check("hnsw recall@10 >= 0.95 at ef=200", rec >= 0.95, f"{rec:.3f}")
check("duplicate live id rejected",
      (lambda: (_raises(lambda: db.add(X[:1], np.array([0], dtype=np.uint64)))))())

# ---------------------------------------------------------------- deletes
print("deletes:")
dead = rng.choice(3000, 600, replace=False).astype(np.uint64)
db.delete(dead)
dead_set = set(dead.tolist())
ids, _ = db.search(Q, k=10, ef=100)
check("no deleted ids in hnsw results",
      all(int(x) not in dead_set for row in ids for x in row))
f_ids, _ = db.search(Q, k=10, exact=True)
check("no deleted ids in flat results",
      all(int(x) not in dead_set for row in f_ids for x in row))
check("len reflects deletes", len(db) == 2400, str(len(db)))
alive_mask = np.ones(3000, bool); alive_mask[dead] = False
amap = np.where(alive_mask)[0]
Da = ((Q[:, None, :] - X[alive_mask][None, :, :]) ** 2).sum(-1)
gta = amap[np.argsort(Da, axis=1)[:, :10]]
reca = np.mean([len(set(a.tolist()) & set(b.tolist()))/10
                for a, b in zip(ids.astype(np.int64), gta)])
check("post-delete hnsw recall >= 0.95", reca >= 0.95, f"{reca:.3f}")
check("missing id delete raises",
      _raises(lambda: db.delete(np.array([10**9], dtype=np.uint64))))
db.delete(np.uint64(amap[0]))
db.add(X[amap[0]][None, :], np.array([amap[0]], dtype=np.uint64))
check("delete-then-readd same id", len(db) == 2400)

# ---------------------------------------------------------------- persistence
print("persistence:")
db.save("/tmp/vecdb_test.vecdb")
db2 = VecDB.load("/tmp/vecdb_test.vecdb")
ids2, _ = db2.search(Q, k=10, ef=100)
ids1, _ = db.search(Q, k=10, ef=100)
check("v2 round-trip identical (tombstones preserved)", np.array_equal(ids1, ids2))
check("loaded len correct", len(db2) == 2400, str(len(db2)))
check("post-load delete works (id map rebuilt)",
      not _raises(lambda: db2.delete(np.uint64(amap[1]))))

# ---------------------------------------------------------------- compact
print("compact:")
db.compact()
ids3, _ = db.search(Q, k=10, ef=100)
rec3 = np.mean([len(set(a.tolist()) & set(b.tolist()))/10
                for a, b in zip(ids3.astype(np.int64), gta)])
check("compact preserves recall >= 0.95", rec3 >= 0.95, f"{rec3:.3f}")
check("compact preserves len", len(db) == 2400)

# ---------------------------------------------------------------- churn
print("churn (3 cycles, 30% turnover, 5k vectors):")
N, DIM, K = 5000, 64, 10
Xc = rng.standard_normal((N, DIM)).astype(np.float32)
dbc = VecDB(dim=DIM); dbc.add(Xc, np.arange(N, dtype=np.uint64))
store = {i: Xc[i] for i in range(N)}
nid = N
worst = 1.0
for _ in range(3):
    kill = rng.choice(list(store.keys()), N*3//10, replace=False)
    dbc.delete(np.array(kill, dtype=np.uint64))
    for k_ in kill: del store[int(k_)]
    newv = rng.standard_normal((len(kill), DIM)).astype(np.float32)
    nids = np.arange(nid, nid+len(kill), dtype=np.uint64); nid += len(kill)
    dbc.add(newv, nids)
    for j, n_ in enumerate(nids): store[int(n_)] = newv[j]
    keys = np.array(list(store.keys()))
    mat = np.stack([store[int(k_)] for k_ in keys])
    Qs = rng.standard_normal((30, DIM)).astype(np.float32)
    Dc = ((Qs[:, None, :] - mat[None, :, :]) ** 2).sum(-1)
    gtc = keys[np.argsort(Dc, axis=1)[:, :K]]
    idc, _ = dbc.search(Qs, k=K, ef=100)
    r = np.mean([len(set(a.tolist()) & set(b.tolist()))/K
                 for a, b in zip(idc.astype(np.int64), gtc)])
    worst = min(worst, r)
check("recall stays >= 0.90 across churn", worst >= 0.90, f"worst {worst:.3f}")
check("live count stable", len(dbc) == N)

# ---------------------------------------------------------------- edge: delete all
small = VecDB(dim=8)
small.add(rng.standard_normal((5, 8)).astype(np.float32))
small.delete(np.arange(5, dtype=np.uint64))
i, d = small.search(rng.standard_normal((1, 8)).astype(np.float32), k=3)
check("delete-all returns 0 results", np.isinf(d).all())

# ---------------------------------------------------------------- turboquant
print("turboquant:")
Xt = rng.standard_normal((2000, 128)).astype(np.float32)
for bits in (4, 8):
    t = TQIndex(128, bits=bits); t.add(Xt)
    ti, _ = t.search(Xt[:20], k=1)
    check(f"{bits}-bit self-NN", (ti[:, 0] == np.arange(20)).all())
t = TQIndex(128, bits=4, qjl=True); t.add(Xt[:500])
check("qjl mode runs", t.search(Xt[:2], k=5)[0].shape == (2, 5))

# ---------------------------------------------------------------- filtered search
print("filtered search:")
Xf = rng.standard_normal((4000, 64)).astype(np.float32)
dbf = VecDB(dim=64); dbf.add(Xf)
Qf = rng.standard_normal((30, 64)).astype(np.float32)
allow = rng.choice(4000, 800, replace=False).astype(np.uint64)
allow_set = set(allow.tolist())
fi, _ = dbf.search(Qf, k=10, ef=200, allow_ids=allow)
check("hnsw allow-list: results subset of allow",
      all(int(x) in allow_set for row in fi for x in row if int(x) != 2**64-1))
ei, _ = dbf.search(Qf, k=10, exact=True, allow_ids=allow)
Df = ((Qf[:, None, :] - Xf[allow][None, :, :]) ** 2).sum(-1)
gtf = allow[np.argsort(Df, axis=1)[:, :10]]
check("exact allow-list matches numpy",
      np.array_equal(ei.astype(np.int64), gtf.astype(np.int64)))
recf = np.mean([len(set(a.tolist()) & set(b.tolist()))/10
                for a, b in zip(fi.astype(np.int64), gtf.astype(np.int64))])
check("hnsw allow-list recall >= 0.9 (20% selectivity)", recf >= 0.9, f"{recf:.3f}")
deny = allow
di, _ = dbf.search(Qf, k=10, ef=200, deny_ids=deny)
check("deny-list: no denied ids in results",
      all(int(x) not in allow_set for row in di for x in row))

# ---------------------------------------------------------------- concurrent search
print("concurrent search:")
import threading
Xs = rng.standard_normal((5000, 64)).astype(np.float32)
dbs = VecDB(dim=64); dbs.add(Xs)
Qss = rng.standard_normal((40, 64)).astype(np.float32)
ref, _ = dbs.search(Qss, k=10, ef=100)
results = [None] * 4
def worker(t_):
    out = [dbs.search(Qss, k=10, ef=100)[0] for _ in range(5)]
    results[t_] = out
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
for th in threads: th.start()
for th in threads: th.join()
ok = all(np.array_equal(r, ref) for outs in results for r in outs)
check("4 threads x 5 searches all match single-threaded reference", ok)

print(f"\n{'ALL TESTS PASS' if FAILS == 0 else f'{FAILS} FAILURES'}")
sys.exit(1 if FAILS else 0)
