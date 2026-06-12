"""pyvecdb — Python bindings for vecdb (the from-scratch C vector database).

Usage:
    from pyvecdb import VecDB
    import numpy as np

    db = VecDB(dim=128, M=16, ef_construction=200)
    vecs = np.random.rand(1000, 128).astype(np.float32)
    db.add(vecs)                          # ids default to 0..n-1
    ids, dists = db.search(vecs[:5], k=10, ef=100)
    db.save("index.vecdb")
    db2 = VecDB.load("index.vecdb")
"""
from __future__ import annotations

import ctypes
import os
from ctypes import (POINTER, Structure, c_char_p, c_float, c_int, c_int64,
                    c_size_t, c_uint64, c_void_p)

import numpy as np

_LIB_PATH = os.environ.get(
    "VECDB_LIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libvecdb.so"),
)
_lib = ctypes.CDLL(_LIB_PATH)


class _VecResult(Structure):
    _fields_ = [("id", c_uint64), ("dist", c_float)]


class _Config(Structure):
    _fields_ = [("dim", c_int), ("M", c_int), ("ef_construction", c_int),
                ("initial_capacity", c_size_t), ("seed", c_uint64)]


_lib.vecdb_create.argtypes = [POINTER(_Config)]
_lib.vecdb_create.restype = c_void_p
_lib.vecdb_free.argtypes = [c_void_p]
_lib.vecdb_add.argtypes = [c_void_p, c_uint64, POINTER(c_float)]
_lib.vecdb_add.restype = c_int64
_lib.vecdb_count.argtypes = [c_void_p]
_lib.vecdb_count.restype = c_size_t
_lib.vecdb_dim.argtypes = [c_void_p]
_lib.vecdb_dim.restype = c_int
_lib.vecdb_search_flat.argtypes = [c_void_p, POINTER(c_float), c_int, POINTER(_VecResult)]
_lib.vecdb_search_flat.restype = c_int
_lib.vecdb_search_flat_batch.argtypes = [c_void_p, POINTER(c_float), c_int, c_int, POINTER(_VecResult)]
_lib.vecdb_search_flat_batch.restype = c_int
_lib.vecdb_search_hnsw.argtypes = [c_void_p, POINTER(c_float), c_int, c_int, POINTER(_VecResult)]
_lib.vecdb_search_hnsw.restype = c_int
_lib.vecdb_search_hnsw_filtered.argtypes = [c_void_p, POINTER(c_float), c_int, c_int,
                                            ctypes.c_char_p, POINTER(_VecResult)]
_lib.vecdb_search_hnsw_filtered.restype = c_int
_lib.vecdb_search_flat_batch_filtered.argtypes = [c_void_p, POINTER(c_float), c_int, c_int,
                                                  ctypes.c_char_p, POINTER(_VecResult)]
_lib.vecdb_search_flat_batch_filtered.restype = c_int
_lib.vecdb_search_hnsw_batch.argtypes = [c_void_p, POINTER(c_float), c_int, c_int,
                                          c_int, ctypes.c_char_p, POINTER(_VecResult)]
_lib.vecdb_search_hnsw_batch.restype = c_int
_lib.vecdb_slots.argtypes = [c_void_p]
_lib.vecdb_slots.restype = c_size_t
_lib.vecdb_make_mask.argtypes = [c_void_p, POINTER(c_uint64), c_size_t, c_int, ctypes.c_char_p]
_lib.vecdb_make_mask.restype = c_int64
_lib.vecdb_save.argtypes = [c_void_p, c_char_p]
_lib.vecdb_save.restype = c_int
_lib.vecdb_load.argtypes = [c_char_p]
_lib.vecdb_load.restype = c_void_p

# Delete support
_lib.vecdb_delete.argtypes = [c_void_p, c_uint64]
_lib.vecdb_delete.restype = c_int
_lib.vecdb_compact.argtypes = [c_void_p]
_lib.vecdb_compact.restype = c_int


def _as_f32_matrix(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"expected shape (n, {dim}), got {x.shape}")
    return x


class VecDB:
    """A vector index with exact (flat) and approximate (HNSW) search."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 capacity: int = 1024, seed: int = 0x9E3779B97F4A7C15,
                 _handle: int | None = None):
        if _handle is not None:
            self._h = _handle
            self.dim = _lib.vecdb_dim(self._h)
            return
        cfg = _Config(dim, M, ef_construction, capacity, seed)
        self._h = _lib.vecdb_create(ctypes.byref(cfg))
        if not self._h:
            raise RuntimeError("vecdb_create failed")
        self.dim = dim

    # -- lifecycle -----------------------------------------------------
    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.vecdb_free(h)
            self._h = None

    def __len__(self) -> int:
        return _lib.vecdb_count(self._h)

    # -- writes --------------------------------------------------------
    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Add a (n, dim) float32 batch. ids default to current_count..+n."""
        vectors = _as_f32_matrix(vectors, self.dim)
        n = vectors.shape[0]
        if ids is None:
            start = len(self)
            ids = np.arange(start, start + n, dtype=np.uint64)
        else:
            ids = np.ascontiguousarray(ids, dtype=np.uint64)
            if ids.shape != (n,):
                raise ValueError("ids must have shape (n,)")
        fp = vectors.ctypes.data_as(POINTER(c_float))
        for i in range(n):
            rc = _lib.vecdb_add(self._h, int(ids[i]),
                                ctypes.cast(ctypes.addressof(fp.contents)
                                            + i * self.dim * 4, POINTER(c_float)))
            if rc < 0:
                raise RuntimeError(f"vecdb_add failed at row {i}")

    def delete(self, ids: np.ndarray) -> None:
        """Delete vectors by their IDs.

        Calls the underlying C ``vecdb_delete`` for each ID. If any deletion fails
        (e.g., the ID is not present) a ``RuntimeError`` is raised.
        """
        ids = np.ascontiguousarray(np.atleast_1d(ids), dtype=np.uint64)
        for i in range(ids.shape[0]):
            rc = _lib.vecdb_delete(self._h, int(ids[i]))
            if rc != 0:
                raise RuntimeError(f"vecdb_delete failed for id {ids[i]}")

    def compact(self) -> None:
        """Rebuild the index without tombstoned nodes. Restores graph
        quality and reclaims memory after many deletes."""
        if _lib.vecdb_compact(self._h) != 0:
            raise RuntimeError("vecdb_compact failed")


    # -- reads ---------------------------------------------------------
    def _make_mask(self, allow_ids, deny_ids) -> bytes | None:
        if allow_ids is None and deny_ids is None:
            return None
        if allow_ids is not None and deny_ids is not None:
            raise ValueError("pass allow_ids or deny_ids, not both")
        ids = np.ascontiguousarray(
            np.atleast_1d(allow_ids if allow_ids is not None else deny_ids),
            dtype=np.uint64)
        mode = 0 if allow_ids is not None else 1
        mask = ctypes.create_string_buffer(_lib.vecdb_slots(self._h))
        _lib.vecdb_make_mask(self._h, ids.ctypes.data_as(POINTER(c_uint64)),
                             ids.shape[0], mode, mask)
        return mask

    def search(self, queries: np.ndarray, k: int = 10, ef: int = 100,
               exact: bool = False, allow_ids=None,
               deny_ids=None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ids, dists), each shaped (nq, k). dist = squared L2.
        Missing slots (k > count) are id=2^64-1, dist=inf.
        allow_ids / deny_ids restrict results to (or away from) the given
        user ids. For very selective allow-lists prefer exact=True."""
        queries = _as_f32_matrix(queries, self.dim)
        nq = queries.shape[0]
        out_ids = np.full((nq, k), np.iinfo(np.uint64).max, dtype=np.uint64)
        out_dists = np.full((nq, k), np.inf, dtype=np.float32)
        qp = queries.ctypes.data_as(POINTER(c_float))
        mask = self._make_mask(allow_ids, deny_ids)
        if exact:                      # single C call, blocked over queries
            buf = (_VecResult * (nq * k))()
            if mask is None:
                n = _lib.vecdb_search_flat_batch(self._h, qp, nq, k, buf)
            else:
                n = _lib.vecdb_search_flat_batch_filtered(self._h, qp, nq, k, mask, buf)
            for i in range(nq):
                for j in range(n):
                    out_ids[i, j] = buf[i * k + j].id
                    out_dists[i, j] = buf[i * k + j].dist
            return out_ids, out_dists
        buf = (_VecResult * (nq * k))()       # one C call, OpenMP inside
        _lib.vecdb_search_hnsw_batch(self._h, qp, nq, k, ef, mask, buf)
        for i in range(nq):
            for j in range(k):
                out_ids[i, j] = buf[i * k + j].id
                out_dists[i, j] = buf[i * k + j].dist
        return out_ids, out_dists

    # -- persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        if _lib.vecdb_save(self._h, path.encode()) != 0:
            raise IOError(f"vecdb_save({path!r}) failed")

    @classmethod
    def load(cls, path: str) -> "VecDB":
        h = _lib.vecdb_load(path.encode())
        if not h:
            raise IOError(f"vecdb_load({path!r}) failed")
        return cls(dim=0, _handle=h)


# ---------------------------------------------------------------------------
# TurboQuant index (core MSE variant of arXiv:2504.19874)
# ---------------------------------------------------------------------------
_lib.tq_create.argtypes = [c_int, c_int, c_uint64]
_lib.tq_create.restype = c_void_p
_lib.tq_create2.argtypes = [c_int, c_int, c_int, c_uint64]
_lib.tq_create2.restype = c_void_p
_lib.tq_free.argtypes = [c_void_p]
_lib.tq_add.argtypes = [c_void_p, c_uint64, POINTER(c_float)]
_lib.tq_add.restype = c_int64
_lib.tq_count.argtypes = [c_void_p]
_lib.tq_count.restype = c_size_t
_lib.tq_search.argtypes = [c_void_p, POINTER(c_float), c_int, POINTER(_VecResult)]
_lib.tq_search.restype = c_int
_lib.tq_search_batch.argtypes = [c_void_p, POINTER(c_float), c_int, c_int, POINTER(_VecResult)]
_lib.tq_search_batch.restype = c_int
_lib.tq_memory_bytes.argtypes = [c_void_p]
_lib.tq_memory_bytes.restype = c_size_t


class TQIndex:
    """TurboQuant compressed index: data-oblivious, no training, online inserts.

    Pipeline per vector: norm separation -> randomized Hadamard rotation
    (3 rounds) -> per-coordinate Lloyd-Max codes against N(0,1).
    Search is an asymmetric brute-force scan over packed codes.
    """

    def __init__(self, dim: int, bits: int = 4, qjl: bool = False, seed: int = 123):
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        self._h = _lib.tq_create2(dim, bits, 1 if qjl else 0, seed)
        if not self._h:
            raise RuntimeError("tq_create failed")
        self.dim, self.bits, self.qjl = dim, bits, qjl

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.tq_free(h)
            self._h = None

    def __len__(self) -> int:
        return _lib.tq_count(self._h)

    def memory_bytes(self) -> int:
        return _lib.tq_memory_bytes(self._h)

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        vectors = _as_f32_matrix(vectors, self.dim)
        n = vectors.shape[0]
        if ids is None:
            start = len(self)
            ids = np.arange(start, start + n, dtype=np.uint64)
        fp = vectors.ctypes.data_as(POINTER(c_float))
        for i in range(n):
            row = ctypes.cast(ctypes.addressof(fp.contents) + i * self.dim * 4,
                              POINTER(c_float))
            if _lib.tq_add(self._h, int(ids[i]), row) < 0:
                raise RuntimeError(f"tq_add failed at row {i}")

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        queries = _as_f32_matrix(queries, self.dim)
        nq = queries.shape[0]
        out_ids = np.zeros((nq, k), dtype=np.uint64)
        out_dists = np.full((nq, k), np.inf, dtype=np.float32)
        buf = (_VecResult * (nq * k))()       # one C call, OpenMP inside
        qp = queries.ctypes.data_as(POINTER(c_float))
        _lib.tq_search_batch(self._h, qp, nq, k, buf)
        for i in range(nq):
            for j in range(k):
                out_ids[i, j] = buf[i * k + j].id
                out_dists[i, j] = buf[i * k + j].dist
        return out_ids, out_dists
