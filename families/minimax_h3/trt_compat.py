# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small family-local compatibility layer for TensorRT Python bindings."""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any


_STANDARD_MODULE = "tensorrt"
_RTX_MODULE = "tensorrt_rtx"
_backend_module_name = _STANDARD_MODULE
_module: ModuleType | None = None


def configure_backend(*, rtx: bool = False) -> None:
    """Select a TensorRT module before any graph-building API is used."""

    global _backend_module_name, _module
    requested = _RTX_MODULE if rtx else _STANDARD_MODULE
    if _module is not None and _backend_module_name != requested:
        raise RuntimeError("a different TensorRT Python module is already loaded")
    if rtx:
        try:
            module = importlib.import_module(_RTX_MODULE)
        except ImportError as error:
            raise ImportError(
                "TensorRT-RTX is required for backend=trt_rtx builds"
            ) from error
        sys.modules[_STANDARD_MODULE] = module
    _backend_module_name = requested


def is_available(module_name: str | None = None) -> bool:
    """Return whether a TensorRT Python module can be imported."""

    name = module_name or _backend_module_name
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def load_module() -> ModuleType:
    """Import and return the selected TensorRT Python module."""

    global _module
    active = sys.modules.get(_backend_module_name)
    if active is not None:
        _module = active
    if _module is None:
        _module = importlib.import_module(_backend_module_name)
    return _module


def get_trt() -> ModuleType:
    return load_module()


def tensorrt_version() -> str:
    return str(getattr(load_module(), "__version__", ""))


def tensorrt_abi(version: str | None = None) -> str:
    match = re.search(r"(\d+)\.(\d+)", version or tensorrt_version())
    return f"{match.group(1)}.{match.group(2)}" if match else ""


def _capsule_byte_view(data: object, size: int) -> memoryview:
    import ctypes

    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = [ctypes.py_object]
    get_name.restype = ctypes.c_char_p
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    name = get_name(data)
    pointer = get_pointer(data, name)
    if not pointer and size:
        raise ValueError("TensorRT stream writer supplied a null data pointer")
    if size == 0:
        return memoryview(b"")
    storage = (ctypes.c_ubyte * size).from_address(pointer or 0)
    return memoryview(storage).cast("B")


def _stream_writer_view(data: object, size: object | None) -> memoryview:
    if size is None:
        source = memoryview(data)
        try:
            return source.cast("B")
        finally:
            source.release()
    count = int(size)
    if count < 0:
        raise ValueError("TensorRT stream writer supplied a negative byte count")
    try:
        source = memoryview(data)
    except TypeError:
        return _capsule_byte_view(data, count)
    try:
        view = source.cast("B")
        if count > view.nbytes:
            raise ValueError("TensorRT stream writer byte count exceeds its source buffer")
        return view[:count]
    finally:
        source.release()


def _file_stream_writer(stream: Any) -> Any:
    module = load_module()
    base = getattr(module, "IStreamWriter", None)
    if base is None:
        raise RuntimeError("TensorRT does not expose direct stream serialization")

    class FileStreamWriter(base):
        def __init__(self) -> None:
            base.__init__(self)
            self.size = 0
            self.error: BaseException | None = None

        def write(self, data: object, size: object | None = None) -> int:
            if self.error is not None:
                return -1
            view: memoryview | None = None
            try:
                view = _stream_writer_view(data, size)
                total = view.nbytes
                for offset in range(0, total, 8 << 20):
                    chunk = view[offset : offset + (8 << 20)]
                    try:
                        written = stream.write(chunk)
                        if written != chunk.nbytes:
                            raise OSError("short TensorRT stream write")
                    finally:
                        chunk.release()
                self.size += total
                return total
            except BaseException as error:
                self.error = error
                return -1
            finally:
                if view is not None:
                    view.release()

    return FileStreamWriter()


def build_serialized_network_to_file(
    builder: Any,
    network: Any,
    config: Any,
    output_path: str | Path,
) -> dict[str, int]:
    """Serialize a network directly to an atomically published plan file."""

    method = getattr(builder, "build_serialized_network_to_stream", None)
    if not callable(method):
        raise RuntimeError("TensorRT direct stream serialization is unavailable")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    writer = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer = _file_stream_writer(stream)
            succeeded = method(network, config, writer)
            if writer.error is not None:
                raise RuntimeError("TensorRT plan stream writer failed") from writer.error
            if not succeeded or writer.size <= 0:
                raise RuntimeError("TensorRT produced no serialized network")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"bytes": writer.size}
