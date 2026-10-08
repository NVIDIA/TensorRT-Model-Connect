# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCCL communicator for multi-rank engine tests (launch with ``tools/launch_ranks.py``).

Select the rank's CUDA device before constructing it (``cudaSetDevice(local_rank)``).

Uses the launcher's contract: ``OMPI_COMM_WORLD_{SIZE,RANK,LOCAL_RANK}``, the unique-id
rendezvous file ``TRTMC_NCCL_RENDEZVOUS`` and the library ``TRTMC_NCCL_LIBRARY``.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


class NcclComm:
    def __init__(self):
        self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self.world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        self.local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", self.rank))
        default = "nccl.dll" if sys.platform.startswith("win") else "libnccl.so.2"
        self.lib = ctypes.CDLL(os.environ.get("TRTMC_NCCL_LIBRARY") or default)
        self.lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        self.lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
        self.lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        p2p_args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.ncclSend.argtypes = p2p_args
        self.lib.ncclRecv.argtypes = p2p_args
        uid = _UniqueId()
        path = Path(os.environ["TRTMC_NCCL_RENDEZVOUS"])
        if self.rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(ctypes.string_at(ctypes.addressof(uid), 128))
            os.replace(tmp, path)
        else:
            deadline = time.monotonic() + 120
            while not path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no NCCL unique id at {path}")
                time.sleep(0.05)
            ctypes.memmove(ctypes.addressof(uid), path.read_bytes(), 128)
        self.comm = ctypes.c_void_p()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), self.world, uid, self.rank),
                    "ncclCommInitRank")
        if self.rank == 0:
            # Every rank joined, so every rank read the id; a reused path must not
            # hand this stale id to the next launch.
            path.unlink(missing_ok=True)

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"{what} failed with NCCL status {status}")

    def capsule(self):
        new = ctypes.pythonapi.PyCapsule_New
        new.restype = ctypes.py_object
        new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return new(self.comm.value, None, None)

    def p2p(self, transfers, stream: int, *, timeout_s: float = 120.0) -> None:
        """One NCCL group of ``(send, device_ptr, nbytes, peer)`` byte transfers on ``stream``.

        Polls the stream with a deadline and aborts the communicator on a timeout, like the
        runtime's ``PeerChannel``.
        """
        from cuda.bindings import runtime as rt

        self._check(self.lib.ncclGroupStart(), "ncclGroupStart")
        for send, ptr, nbytes, peer in transfers:
            op = self.lib.ncclSend if send else self.lib.ncclRecv
            self._check(op(ctypes.c_void_p(ptr), nbytes, 1, peer, self.comm, ctypes.c_void_p(stream)),
                        "ncclSend" if send else "ncclRecv")
        self._check(self.lib.ncclGroupEnd(), "ncclGroupEnd")
        deadline = time.monotonic() + timeout_s
        while int(rt.cudaStreamQuery(stream)[0]) != 0:
            if time.monotonic() > deadline:
                self.abort()
                raise TimeoutError(f"NCCL transfers did not finish within {timeout_s} s; communicator aborted")
            time.sleep(0.002)

    def abort(self) -> None:
        """``ncclCommAbort``: makes in-flight NCCL kernels exit (use instead of killing a hung rank)."""
        if self.comm:
            self.lib.ncclCommAbort(self.comm)
            self.comm = ctypes.c_void_p()

    def destroy(self) -> None:
        if self.comm:
            self.lib.ncclCommDestroy(self.comm)
            self.comm = ctypes.c_void_p()
