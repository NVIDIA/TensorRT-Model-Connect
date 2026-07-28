# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural validation for tokenizer artifacts consumed by native runtimes."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

_FLOAT32_MAX = 3.4028234663852886e38
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_NATIVE_MODEL_TYPES = {"BPE", "WordPiece", "Unigram"}
_TOKENIZER_REPAIR_LOCK_NAME = ".trtmc-tokenizer-repair.lock"

# A directory inode is the shared ownership boundary for every tokenizer.json
# repair path. The process-local lock serializes threads (and makes nested
# outer -> family repairs reentrant). A persistent regular-file sentinel gives
# flock a writable descriptor on NFS and is never unlinked, avoiding stale
# lock-file inode races after crashes.
_TOKENIZER_REPAIR_LOCKS_GUARD = threading.Lock()
_TOKENIZER_REPAIR_LOCKS: dict[tuple[int, int], threading.RLock] = {}
_TOKENIZER_REPAIR_LOCK_DEPTH = threading.local()
_TOKENIZER_REPAIR_FDS: set[int] = set()
_TOKENIZER_REPAIR_FORK_GUARD = threading.RLock()


def _tokenizer_repair_atfork_hook(phase: str) -> None:
    """Test seam for deterministic fork-window probes."""
    del phase


def _tokenizer_repair_fd_lifecycle_hook(
    phase: str,
    descriptor: int,
) -> None:
    """Test seam around FD registration and release critical sections."""
    del phase, descriptor


def _before_tokenizer_repair_fork() -> None:
    _tokenizer_repair_atfork_hook("before")
    _TOKENIZER_REPAIR_FORK_GUARD.acquire()


def _after_tokenizer_repair_fork_in_parent() -> None:
    try:
        _tokenizer_repair_atfork_hook("parent")
    finally:
        _TOKENIZER_REPAIR_FORK_GUARD.release()


def _after_tokenizer_repair_fork_in_child() -> None:
    """Close inherited transaction FDs and reset thread locks in a child."""
    global _TOKENIZER_REPAIR_LOCKS_GUARD
    global _TOKENIZER_REPAIR_LOCKS
    global _TOKENIZER_REPAIR_LOCK_DEPTH
    global _TOKENIZER_REPAIR_FDS
    global _TOKENIZER_REPAIR_FORK_GUARD

    # Closing the child's copies does not release the parent's flock; it keeps
    # the child from mistaking an inherited open-file-description lock for its
    # own reentrant acquisition.
    for descriptor in tuple(_TOKENIZER_REPAIR_FDS):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _TOKENIZER_REPAIR_LOCKS_GUARD = threading.Lock()
    _TOKENIZER_REPAIR_LOCKS = {}
    _TOKENIZER_REPAIR_LOCK_DEPTH = threading.local()
    _TOKENIZER_REPAIR_FDS = set()
    _TOKENIZER_REPAIR_FORK_GUARD = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_tokenizer_repair_fork,
        after_in_parent=_after_tokenizer_repair_fork_in_parent,
        after_in_child=_after_tokenizer_repair_fork_in_child,
    )


def _acquire_repair_flock(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _release_repair_flock(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def tokenizer_repair_lock_present(model_dir: str | Path) -> bool:
    """Return whether callers must synchronize before trusting canonical data."""
    sentinel = Path(model_dir) / _TOKENIZER_REPAIR_LOCK_NAME
    try:
        sentinel.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An inaccessible or otherwise abnormal sentinel must never enable the
        # unlocked fast path.
        return True
    return True


def _open_registered_repair_descriptor(
    path: str | Path,
    flags: int,
    *,
    mode: int | None = None,
    dir_fd: int | None = None,
) -> int:
    """Open and register an FD atomically with respect to process fork."""
    with _TOKENIZER_REPAIR_FORK_GUARD:
        if mode is None:
            descriptor = os.open(path, flags)
        else:
            descriptor = os.open(path, flags, mode, dir_fd=dir_fd)
        _tokenizer_repair_fd_lifecycle_hook(
            "opened-before-register",
            descriptor,
        )
        _TOKENIZER_REPAIR_FDS.add(descriptor)
        return descriptor


def _close_registered_repair_descriptor(descriptor: int | None) -> None:
    """Close before unregistering while process fork is excluded."""
    if descriptor is None:
        return
    with _TOKENIZER_REPAIR_FORK_GUARD:
        try:
            os.close(descriptor)
        except OSError:
            pass
        finally:
            _TOKENIZER_REPAIR_FDS.discard(descriptor)


@contextmanager
def tokenizer_repair_lock(model_dir: str | Path) -> Iterator[None]:
    """Own tokenizer repair for one model directory across threads/processes.

    The directory and sentinel descriptors remain open for the complete outer
    transaction. Nested acquisition by the owning thread is process-local
    only, avoiding a second open/flock that could deadlock while preserving the
    outer cross-process lock. A forked child drops inherited ownership in the
    at-fork callback; exiting an inherited lexical context in that child is a
    no-op, and fresh child repair ownership must be acquired separately.
    """
    acquisition_pid = os.getpid()
    directory = Path(model_dir)
    try:
        canonical_directory = str(directory.resolve(strict=True))
    except Exception as exc:
        raise RuntimeError(
            "cannot acquire tokenizer.json repair ownership for "
            f"'{directory}' before modifying it: {exc}"
        ) from exc

    depths = getattr(_TOKENIZER_REPAIR_LOCK_DEPTH, "depths", None)
    held_paths = getattr(_TOKENIZER_REPAIR_LOCK_DEPTH, "held_paths", None)
    if depths is None or held_paths is None:
        depths = {}
        held_paths = {}
        _TOKENIZER_REPAIR_LOCK_DEPTH.depths = depths
        _TOKENIZER_REPAIR_LOCK_DEPTH.held_paths = held_paths

    # The normal outer -> standard -> family path reuses the already-open
    # directory descriptor and flock. resolve() also makes symlink aliases hit
    # this fast path.
    nested_key = held_paths.get(canonical_directory)
    if nested_key is not None:
        with _TOKENIZER_REPAIR_LOCKS_GUARD:
            nested_lock = _TOKENIZER_REPAIR_LOCKS[nested_key]
        nested_lock.acquire()
        depths[nested_key] += 1
        try:
            yield
        finally:
            if os.getpid() == acquisition_pid:
                depths[nested_key] -= 1
                nested_lock.release()
        return

    directory_descriptor: int | None = None
    sentinel_descriptor: int | None = None
    flock_acquired = False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directory_descriptor = _open_registered_repair_descriptor(
            directory,
            flags,
        )
        metadata = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(f"not a directory: '{directory}'")
    except Exception as exc:
        if os.getpid() == acquisition_pid:
            _close_registered_repair_descriptor(directory_descriptor)
        directory_descriptor = None
        raise RuntimeError(
            "cannot acquire tokenizer.json repair ownership for "
            f"'{directory}' before modifying it: {exc}"
        ) from exc

    lock_key = (metadata.st_dev, metadata.st_ino)
    with _TOKENIZER_REPAIR_LOCKS_GUARD:
        local_lock = _TOKENIZER_REPAIR_LOCKS.setdefault(
            lock_key,
            threading.RLock(),
        )

    acquired_local = False
    try:
        try:
            local_lock.acquire()
            acquired_local = True
        except Exception as exc:
            raise RuntimeError(
                "cannot acquire process-local tokenizer.json repair "
                f"ownership for '{directory}': {exc}"
            ) from exc

        depth = depths.get(lock_key, 0)
        if depth:
            depths[lock_key] = depth + 1
            held_paths[canonical_directory] = lock_key
            _close_registered_repair_descriptor(directory_descriptor)
            directory_descriptor = None
            try:
                yield
            finally:
                if os.getpid() == acquisition_pid:
                    nested_depth = depths[lock_key] - 1
                    if nested_depth:
                        depths[lock_key] = nested_depth
                    else:
                        depths.pop(lock_key, None)
                    if held_paths.get(canonical_directory) == lock_key:
                        held_paths.pop(canonical_directory, None)
            return

        try:
            required_flags = ("O_NOFOLLOW", "O_NONBLOCK")
            missing_flags = [
                name for name in required_flags if not hasattr(os, name)
            ]
            if missing_flags:
                raise OSError(
                    "secure tokenizer repair locking requires "
                    + ", ".join(missing_flags)
                )
            sentinel_flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_CLOEXEC
                | os.O_NOFOLLOW
                | os.O_NONBLOCK
            )
            sentinel_descriptor = _open_registered_repair_descriptor(
                _TOKENIZER_REPAIR_LOCK_NAME,
                sentinel_flags,
                mode=0o600,
                dir_fd=directory_descriptor,
            )
            sentinel_metadata = os.fstat(sentinel_descriptor)
            if (
                not stat.S_ISREG(sentinel_metadata.st_mode)
                or sentinel_metadata.st_nlink != 1
            ):
                raise OSError(
                    "tokenizer repair sentinel must be a regular file "
                    "with exactly one link"
                )
            _acquire_repair_flock(sentinel_descriptor)
            flock_acquired = True
            visible_metadata = os.stat(
                _TOKENIZER_REPAIR_LOCK_NAME,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            visible_directory_metadata = os.stat(
                directory,
                follow_symlinks=True,
            )
            if (
                not stat.S_ISREG(visible_metadata.st_mode)
                or visible_metadata.st_nlink != 1
                or visible_metadata.st_dev != sentinel_metadata.st_dev
                or visible_metadata.st_ino != sentinel_metadata.st_ino
            ):
                raise OSError(
                    "tokenizer repair sentinel changed while acquiring its lock"
                )
            if (
                not stat.S_ISDIR(visible_directory_metadata.st_mode)
                or visible_directory_metadata.st_dev != metadata.st_dev
                or visible_directory_metadata.st_ino != metadata.st_ino
            ):
                raise OSError(
                    "model directory changed while acquiring tokenizer "
                    "repair ownership"
                )
        except Exception as exc:
            raise RuntimeError(
                "cannot acquire cross-process tokenizer.json repair "
                f"ownership for '{directory}' before modifying it: {exc}"
            ) from exc

        depths[lock_key] = 1
        held_paths[canonical_directory] = lock_key
        try:
            yield
        finally:
            if os.getpid() == acquisition_pid:
                depths.pop(lock_key, None)
                if held_paths.get(canonical_directory) == lock_key:
                    held_paths.pop(canonical_directory, None)
    finally:
        # Keep both descriptors registered until the flock is explicitly
        # released and both closes have completed. The at-fork before callback
        # holds the same guard, eliminating acquire/register and
        # unregister/unlock inheritance windows.
        if os.getpid() == acquisition_pid:
            with _TOKENIZER_REPAIR_FORK_GUARD:
                if sentinel_descriptor is not None:
                    _tokenizer_repair_fd_lifecycle_hook(
                        "before-unlock-close",
                        sentinel_descriptor,
                    )
                    if flock_acquired:
                        try:
                            _release_repair_flock(sentinel_descriptor)
                        except Exception as exc:
                            # Closing the descriptor still releases flock. The
                            # transaction has already committed or rolled back,
                            # so do not turn success into a false failure.
                            try:
                                print(
                                    "[trtmc build] tokenizer.json repair "
                                    "completed, but explicit lock release "
                                    "failed; descriptor close will release it "
                                    f"for '{directory}': {exc}",
                                    file=sys.stderr,
                                )
                            except Exception:
                                pass
                    try:
                        os.close(sentinel_descriptor)
                    except OSError:
                        pass
                    finally:
                        _TOKENIZER_REPAIR_FDS.discard(sentinel_descriptor)
                    sentinel_descriptor = None
                if directory_descriptor is not None:
                    try:
                        os.close(directory_descriptor)
                    except OSError:
                        pass
                    finally:
                        _TOKENIZER_REPAIR_FDS.discard(directory_descriptor)
                    directory_descriptor = None
            if acquired_local:
                local_lock.release()


def _is_int32(value: object) -> bool:
    return type(value) is int and _INT32_MIN <= value <= _INT32_MAX


def _is_finite_float32_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and abs(value) <= _FLOAT32_MAX


def _optional_field_error(
    node: dict[str, Any],
    key: str,
    *,
    predicate: Callable[[object], bool],
    expected: str,
    path: str,
    allow_null: bool = False,
) -> str | None:
    if key not in node:
        return None
    value = node[key]
    if allow_null and value is None:
        return None
    if not predicate(value):
        return f"{path}.{key} must be {expected}"
    return None


def _string_field_error(
    node: dict[str, Any],
    key: str,
    path: str,
    *,
    allow_null: bool = False,
) -> str | None:
    return _optional_field_error(
        node,
        key,
        predicate=lambda value: isinstance(value, str),
        expected="a string",
        path=path,
        allow_null=allow_null,
    )


def _bool_field_error(
    node: dict[str, Any],
    key: str,
    path: str,
    *,
    allow_null: bool = False,
) -> str | None:
    return _optional_field_error(
        node,
        key,
        predicate=lambda value: type(value) is bool,
        expected="a boolean",
        path=path,
        allow_null=allow_null,
    )


def _int32_field_error(
    node: dict[str, Any],
    key: str,
    path: str,
) -> str | None:
    return _optional_field_error(
        node,
        key,
        predicate=_is_int32,
        expected="a signed 32-bit integer",
        path=path,
    )


def _typed_section(
    document: dict[str, Any],
    key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if key not in document or document[key] is None:
        return None, None
    section = document[key]
    if not isinstance(section, dict):
        return None, f"tokenizer.json {key} must be an object or null"
    error = _string_field_error(section, "type", key)
    if error:
        return None, error
    return section, None


def _object_array(
    node: dict[str, Any],
    key: str,
    path: str,
    *,
    allow_null: bool = False,
) -> tuple[list[Any] | None, str | None]:
    if key not in node:
        return None, None
    values = node[key]
    if allow_null and values is None:
        return None, None
    if not isinstance(values, list):
        return None, f"{path}.{key} must be an array"
    return values, None


def _has_unpaired_surrogate(value: str) -> bool:
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 >= len(value):
                return True
            next_codepoint = ord(value[index + 1])
            if not 0xDC00 <= next_codepoint <= 0xDFFF:
                return True
            index += 2
            continue
        if 0xDC00 <= codepoint <= 0xDFFF:
            return True
        index += 1
    return False


def _invalid_json_scalar_error(
    value: object,
    path: str = "tokenizer.json",
) -> str | None:
    pending: list[tuple[object, str]] = [(value, path)]
    while pending:
        current, current_path = pending.pop()
        if type(current) is float and not math.isfinite(current):
            return (
                f"{current_path} contains a non-finite or JSON-overflow number"
            )
        if type(current) is int:
            try:
                native_float = float(current)
            except OverflowError:
                return (
                    f"{current_path} contains an integer outside the native "
                    "JSON number envelope"
                )
            if not math.isfinite(native_float):
                return (
                    f"{current_path} contains an integer outside the native "
                    "JSON number envelope"
                )
        if isinstance(current, str) and _has_unpaired_surrogate(current):
            return f"{current_path} contains an unpaired UTF-16 surrogate"
        if isinstance(current, dict):
            for key, child in reversed(tuple(current.items())):
                if _has_unpaired_surrogate(key):
                    return (
                        f"{current_path} contains a key with an unpaired "
                        "UTF-16 surrogate"
                    )
                pending.append((child, f"{current_path}[{key!r}]"))
        elif isinstance(current, list):
            for index in range(len(current) - 1, -1, -1):
                pending.append((current[index], f"{current_path}[{index}]"))
    return None


def _added_tokens_error(
    document: dict[str, Any],
    model_type: str,
    base_vocab_size: int,
) -> str | None:
    if "added_tokens" not in document:
        return None
    added_tokens = document["added_tokens"]
    if not isinstance(added_tokens, list):
        return "tokenizer.json added_tokens must be an array"
    required = model_type in {"BPE", "WordPiece"}
    for index, token in enumerate(added_tokens):
        path = f"added_tokens[{index}]"
        if not isinstance(token, dict):
            return f"{path} must be an object"
        for key, predicate, expected in (
            ("content", lambda value: isinstance(value, str), "a string"),
            ("id", _is_int32, "a signed 32-bit integer"),
        ):
            if required and key not in token:
                return f"{path}.{key} is required"
            if key in token and not predicate(token[key]):
                return f"{path}.{key} must be {expected}"
        if model_type in {"BPE", "WordPiece"}:
            error = _bool_field_error(token, "special", path)
            if error:
                return error
        token_id = token.get("id", -1)
        content = token.get("content", "")
        if required and token_id < 0:
            return f"{path}.id must be non-negative"
        resizes_native_vocab = (
            token_id >= 0
            and (model_type in {"BPE", "WordPiece"} or bool(content))
        )
        max_contiguous_id = base_vocab_size + len(added_tokens) - 1
        if resizes_native_vocab and token_id > max_contiguous_id:
            return (
                f"{path}.id={token_id} exceeds the contiguous native vocabulary "
                f"allocation bound {max_contiguous_id}"
            )
    return None


def _roberta_pair_ids_error(section: dict[str, Any], path: str) -> str | None:
    for key in ("cls", "sep"):
        if key not in section:
            continue
        pair = section[key]
        if isinstance(pair, list) and len(pair) >= 2 and not _is_int32(pair[1]):
            return f"{path}.{key}[1] must be a signed 32-bit integer"
    return None


def _template_single_error(
    section: dict[str, Any],
    path: str,
    *,
    iterate_object_values: bool = False,
) -> str | None:
    single = section.get("single")
    if isinstance(single, list):
        entries = single
    elif iterate_object_values and isinstance(single, dict):
        entries = list(single.values())
    else:
        return None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if "Sequence" in entry:
            continue
        if "SpecialToken" not in entry:
            continue
        special = entry["SpecialToken"]
        special_path = f"{path}.single[{index}].SpecialToken"
        if not isinstance(special, dict):
            return f"{special_path} must be an object"
        error = _string_field_error(special, "id", special_path)
        if error:
            return error
    return None


def _bpe_digit_group_error(pattern: str, path: str) -> str | None:
    marker_position = pattern.find(r"\p{N}{")
    if marker_position < 0:
        return None
    number_start = marker_position + len(r"\p{N}{")
    close = pattern.find("}", number_start)
    if close < 0:
        return None
    comma = pattern.find(",", number_start)
    if comma < 0 or comma >= close:
        return None
    candidate = pattern[comma + 1 : close]
    match = re.match(r"^[ \t\n\r\f\v]*([+-]?[0-9]+)", candidate, re.ASCII)
    if match is None:
        return f"{path} has a digit-group bound that native std::stoi cannot parse"
    try:
        value = int(match.group(1))
    except ValueError:
        return f"{path} has a digit-group bound outside signed 32-bit range"
    if not _INT32_MIN <= value <= _INT32_MAX:
        return f"{path} has a digit-group bound outside signed 32-bit range"
    return None


def _bpe_split_error(
    split: dict[str, Any],
    path: str,
    *,
    direct: bool,
) -> str | None:
    pattern = split.get("pattern")
    if not isinstance(pattern, dict):
        if direct:
            return (
                f"{path} is unsupported by the native BPE tokenizer; direct "
                'Split requires pattern.String == " "'
            )
        return None
    if direct:
        error = _string_field_error(pattern, "String", f"{path}.pattern")
        if error:
            return error
        if pattern.get("String") != " ":
            return (
                f"{path} is unsupported by the native BPE tokenizer; direct "
                'Split requires pattern.String == " "'
            )
        return None
    if "Regex" in pattern:
        error = _string_field_error(pattern, "Regex", f"{path}.pattern")
        if error:
            return error
        regex = pattern["Regex"]
        if "[^\r\n" in regex or r"[^\r\n" in regex:
            error = _bpe_digit_group_error(
                regex,
                f"{path}.pattern.Regex",
            )
            if error:
                return error
    return None


def _bpe_decoder_replace_error(
    replace: dict[str, Any],
    path: str,
) -> str | None:
    pattern = replace.get("pattern")
    if isinstance(pattern, dict) and "String" in pattern:
        error = _string_field_error(pattern, "String", f"{path}.pattern")
        if error:
            return error
    return _string_field_error(replace, "content", path)


def _bpe_normalizer_replace_error(
    replace: dict[str, Any],
    path: str,
) -> str | None:
    pattern = replace.get("pattern")
    if not isinstance(pattern, dict) or "String" not in pattern:
        return None
    error = _string_field_error(pattern, "String", f"{path}.pattern")
    if error:
        return error
    if pattern["String"] != " ":
        return None
    return _string_field_error(replace, "content", path)


def _bpe_normalizer_replace_scan(
    normalizer: dict[str, Any],
    path: str,
) -> tuple[bool, str | None]:
    error = _string_field_error(normalizer, "type", path)
    if error:
        return False, error
    normalizer_type = normalizer.get("type", "")
    if normalizer_type == "Replace":
        error = _bpe_normalizer_replace_error(normalizer, path)
        if error:
            return False, error
        pattern = normalizer.get("pattern")
        matched = (
            isinstance(pattern, dict)
            and pattern.get("String") == " "
            and normalizer.get("content", "") == "▁"
        )
        return matched, None
    if normalizer_type != "Sequence":
        return False, None
    children, error = _object_array(
        normalizer,
        "normalizers",
        path,
        allow_null=True,
    )
    if error or children is None:
        return False, error
    for index, child in enumerate(children):
        child_path = f"{path}.normalizers[{index}]"
        if not isinstance(child, dict):
            return False, f"{child_path} must be an object"
        matched, error = _bpe_normalizer_replace_scan(
            child,
            child_path,
        )
        if error:
            return False, error
        if matched:
            return True, None
    return False, None


def _bpe_normalizer_error(document: dict[str, Any]) -> str | None:
    normalizer, error = _typed_section(document, "normalizer")
    if error or normalizer is None:
        return error
    if normalizer.get("type", "") == "Sequence":
        children, error = _object_array(
            normalizer,
            "normalizers",
            "normalizer",
            allow_null=True,
        )
        if error or children is None:
            return error
        for index, child in enumerate(children):
            path = f"normalizer.normalizers[{index}]"
            if not isinstance(child, dict):
                return f"{path} must be an object"
            error = _string_field_error(child, "type", path)
            if error:
                return error
            if child.get("type", "") == "Prepend":
                return None
    _, error = _bpe_normalizer_replace_scan(normalizer, "normalizer")
    return error


def _bpe_pre_tokenizer_error(document: dict[str, Any]) -> str | None:
    pre_tokenizer, error = _typed_section(document, "pre_tokenizer")
    if error or pre_tokenizer is None:
        return error
    pre_type = pre_tokenizer.get("type", "")
    if pre_type == "Split":
        return _bpe_split_error(pre_tokenizer, "pre_tokenizer", direct=True)
    if pre_type == "Sequence":
        children, error = _object_array(
            pre_tokenizer,
            "pretokenizers",
            "pre_tokenizer",
            allow_null=True,
        )
        if error or children is None:
            return error
        for index, child in enumerate(children):
            child_path = f"pre_tokenizer.pretokenizers[{index}]"
            if not isinstance(child, dict):
                return f"{child_path} must be an object"
            error = _string_field_error(child, "type", child_path)
            if error:
                return error
            if child.get("type", "") == "Split":
                error = _bpe_split_error(child, child_path, direct=False)
                if error:
                    return error
                pattern = child.get("pattern")
                if isinstance(pattern, dict) and "Regex" in pattern:
                    break
        return None
    if pre_type not in {"", "ByteLevel", "Metaspace"}:
        return f"unsupported native BPE pre_tokenizer.type: {pre_type!r}"
    return None


def _bpe_decoder_error(document: dict[str, Any]) -> str | None:
    decoder, error = _typed_section(document, "decoder")
    if error or decoder is None or decoder.get("type", "") != "Sequence":
        return error
    children, error = _object_array(
        decoder,
        "decoders",
        "decoder",
        allow_null=True,
    )
    if error or children is None:
        return error
    for index, child in enumerate(children):
        path = f"decoder.decoders[{index}]"
        if not isinstance(child, dict):
            return f"{path} must be an object"
        error = _string_field_error(child, "type", path)
        if error:
            return error
        child_type = child.get("type", "")
        if child_type == "Replace":
            error = _bpe_decoder_replace_error(child, path)
        elif child_type == "Strip":
            error = _string_field_error(child, "content", path)
            if not error and child.get("content", " ") == " ":
                error = _int32_field_error(child, "start", path)
        if error:
            return error
    return None


def _bpe_post_processor_error(document: dict[str, Any]) -> str | None:
    processor, error = _typed_section(document, "post_processor")
    if error or processor is None:
        return error
    processor_type = processor.get("type", "")
    if processor_type == "TemplateProcessing":
        return _template_single_error(
            processor,
            "post_processor",
            iterate_object_values=True,
        )
    if processor_type == "RobertaProcessing":
        return _roberta_pair_ids_error(processor, "post_processor")
    if processor_type != "Sequence":
        return None
    children, error = _object_array(
        processor,
        "processors",
        "post_processor",
        allow_null=True,
    )
    if error or children is None:
        return error
    for index, child in enumerate(children):
        path = f"post_processor.processors[{index}]"
        if not isinstance(child, dict):
            return f"{path} must be an object"
        error = _string_field_error(child, "type", path)
        if error:
            return error
        if child.get("type", "") == "TemplateProcessing":
            error = _template_single_error(
                child,
                path,
                iterate_object_values=True,
            )
            if error:
                return error
            break
    return None


def _bpe_error(document: dict[str, Any], model: dict[str, Any]) -> str | None:
    error = _bool_field_error(model, "byte_fallback", "model")
    if error:
        return error
    vocab = model.get("vocab")
    error = _object_vocab_error(vocab, "BPE")
    if error:
        return error
    merges = model.get("merges")
    if not isinstance(merges, list):
        return "BPE model.merges must be an array"
    for index, merge in enumerate(merges):
        if isinstance(merge, str):
            continue
        if (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(token, str) for token in merge)
        ):
            continue
        return (
            f"BPE model.merges entries must be strings or two-string arrays (invalid entry {index})"
        )
    for validator in (
        _bpe_pre_tokenizer_error,
        _bpe_normalizer_error,
        _bpe_decoder_error,
        _bpe_post_processor_error,
    ):
        error = validator(document)
        if error:
            return error
    return None


def _object_vocab_error(vocab: object, model_type: str) -> str | None:
    if not isinstance(vocab, dict) or not vocab:
        return f"{model_type} model.vocab must be a non-empty object"
    token_ids = list(vocab.values())
    if any(type(token_id) is not int for token_id in token_ids):
        return f"{model_type} model.vocab IDs must be integers"
    if len(set(token_ids)) != len(token_ids):
        return f"{model_type} model.vocab IDs must be unique"
    if set(token_ids) != set(range(len(token_ids))):
        return f"{model_type} model.vocab IDs must cover 0..{len(token_ids) - 1}"
    return None


def _wordpiece_bert_normalizer_error(
    normalizer: dict[str, Any],
    path: str,
) -> str | None:
    for key in ("clean_text", "handle_chinese_chars", "lowercase"):
        error = _bool_field_error(normalizer, key, path)
        if error:
            return error
    return _bool_field_error(
        normalizer,
        "strip_accents",
        path,
        allow_null=True,
    )


def _wordpiece_normalizer_error(document: dict[str, Any]) -> str | None:
    normalizer, error = _typed_section(document, "normalizer")
    if error or normalizer is None:
        return error
    normalizer_type = normalizer.get("type", "")
    if normalizer_type == "BertNormalizer":
        return _wordpiece_bert_normalizer_error(normalizer, "normalizer")
    if normalizer_type != "Sequence":
        return None
    children, error = _object_array(
        normalizer,
        "normalizers",
        "normalizer",
        allow_null=True,
    )
    if error or children is None:
        return error
    for index, child in enumerate(children):
        path = f"normalizer.normalizers[{index}]"
        if not isinstance(child, dict):
            return f"{path} must be an object"
        error = _string_field_error(child, "type", path)
        if error:
            return error
        if child.get("type", "") == "BertNormalizer":
            error = _wordpiece_bert_normalizer_error(child, path)
            if error:
                return error
    return None


def _wordpiece_post_processor_error(document: dict[str, Any]) -> str | None:
    processor, error = _typed_section(document, "post_processor")
    if error or processor is None:
        return error
    if processor.get("type", "") not in {
        "TemplateProcessing",
        "BertProcessing",
        "RobertaProcessing",
    }:
        return None
    return _roberta_pair_ids_error(processor, "post_processor")


def _wordpiece_error(
    document: dict[str, Any],
    model: dict[str, Any],
) -> str | None:
    for key in ("unk_token", "continuing_subword_prefix"):
        error = _string_field_error(model, key, "model")
        if error:
            return error
    error = _int32_field_error(model, "max_input_chars_per_word", "model")
    if error:
        return error
    error = _object_vocab_error(model.get("vocab"), "WordPiece")
    if error:
        return error
    for validator in (
        _wordpiece_normalizer_error,
        _wordpiece_post_processor_error,
    ):
        error = validator(document)
        if error:
            return error
    return None


def _unigram_normalizer_error(document: dict[str, Any]) -> str | None:
    normalizer, error = _typed_section(document, "normalizer")
    if error or normalizer is None or normalizer.get("type", "") != "Sequence":
        return error
    children, error = _object_array(
        normalizer,
        "normalizers",
        "normalizer",
        allow_null=True,
    )
    if error or children is None:
        return error
    for index, child in enumerate(children):
        path = f"normalizer.normalizers[{index}]"
        if not isinstance(child, dict):
            return f"{path} must be an object"
        error = _string_field_error(
            child,
            "type",
            path,
        )
        if error:
            return error
    return None


def _unigram_pre_tokenizer_error(document: dict[str, Any]) -> str | None:
    pre_tokenizer, error = _typed_section(document, "pre_tokenizer")
    if error or pre_tokenizer is None:
        return error
    pre_type = pre_tokenizer.get("type", "")
    if pre_type == "Metaspace":
        return _bool_field_error(
            pre_tokenizer,
            "add_prefix_space",
            "pre_tokenizer",
        )
    if pre_type != "Sequence":
        return None
    children, error = _object_array(
        pre_tokenizer,
        "pretokenizers",
        "pre_tokenizer",
        allow_null=True,
    )
    if error or children is None:
        return error
    for index, child in enumerate(children):
        path = f"pre_tokenizer.pretokenizers[{index}]"
        if not isinstance(child, dict):
            return f"{path} must be an object"
        error = _string_field_error(child, "type", path)
        if error:
            return error
        if child.get("type", "") == "Metaspace":
            error = _bool_field_error(child, "add_prefix_space", path)
            if error:
                return error
            break
    return None


def _unigram_post_processor_error(document: dict[str, Any]) -> str | None:
    processor, error = _typed_section(document, "post_processor")
    if error or processor is None:
        return error
    processor_type = processor.get("type", "")
    if processor_type == "TemplateProcessing":
        return _template_single_error(processor, "post_processor")
    if processor_type == "RobertaProcessing":
        return _roberta_pair_ids_error(processor, "post_processor")
    return None


def _unigram_error(
    document: dict[str, Any],
    model: dict[str, Any],
) -> str | None:
    vocab = model.get("vocab")
    if not isinstance(vocab, list) or not vocab:
        return "Unigram model.vocab must be a non-empty array"
    for index, entry in enumerate(vocab):
        if (
            not isinstance(entry, list)
            or len(entry) < 2
            or not isinstance(entry[0], str)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], (int, float))
        ):
            return (
                "Unigram model.vocab entries must contain a string token and "
                f"numeric score (invalid entry {index})"
            )
        if not _is_finite_float32_number(entry[1]):
            return (
                f"Unigram model.vocab scores must be finite float32 numbers (invalid entry {index})"
            )
    unk_id = model.get("unk_id", 0)
    if not _is_int32(unk_id) or not 0 <= unk_id < len(vocab):
        return "Unigram model.unk_id must be a signed 32-bit index into model.vocab"
    for validator in (
        _unigram_normalizer_error,
        _unigram_pre_tokenizer_error,
        _unigram_post_processor_error,
    ):
        error = validator(document)
        if error:
            return error
    return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _read_regular_utf8_file(path: Path) -> tuple[str | None, str | None]:
    """Read a path without blocking and only after its target is a regular file."""
    descriptor: int | None = None
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None, "tokenizer.json must resolve to a regular file"
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = None
            return stream.read(), None
    except UnicodeError as exc:
        return None, f"cannot decode tokenizer.json as UTF-8: {exc}"
    except OSError as exc:
        return None, f"cannot read tokenizer.json: {exc}"
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def native_tokenizer_json_error(tokenizer_path: Path) -> str | None:
    """Return why ``tokenizer.json`` is incompatible with native tokenizers.

    The validation mirrors every JSON field that the native BPE, WordPiece,
    and Unigram constructors type-convert. Unknown fields that those
    constructors safely ignore remain allowed.
    """
    raw_document, read_error = _read_regular_utf8_file(tokenizer_path)
    if read_error is not None:
        return read_error
    assert raw_document is not None
    try:
        document = json.loads(
            raw_document,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return f"invalid tokenizer.json: {exc}"
    except RecursionError:
        return "invalid tokenizer.json: nesting exceeds the JSON decoder limit"

    scalar_error = _invalid_json_scalar_error(document)
    if scalar_error:
        return scalar_error
    if not isinstance(document, dict):
        return "tokenizer.json root must be an object"
    model = document.get("model")
    if not isinstance(model, dict):
        return "tokenizer.json must contain an object-valued model"

    if "type" in model:
        model_type = model["type"]
        if not isinstance(model_type, str):
            return "tokenizer.json model.type must be a string when present"
        if model_type not in _NATIVE_MODEL_TYPES:
            return f"unsupported tokenizer model.type: {model_type!r}"
    else:
        vocab = model.get("vocab")
        if isinstance(vocab, dict) and isinstance(model.get("merges"), list):
            model_type = "BPE"
        elif (
            isinstance(vocab, dict)
            and "merges" not in model
            and "continuing_subword_prefix" in model
        ):
            model_type = "WordPiece"
        elif (
            isinstance(vocab, list)
            and "merges" not in model
            and "continuing_subword_prefix" not in model
        ):
            model_type = "Unigram"
        else:
            return "cannot identify a native BPE, WordPiece, or Unigram model"

    try:
        if model_type == "BPE":
            error = _bpe_error(document, model)
        elif model_type == "WordPiece":
            error = _wordpiece_error(document, model)
        else:
            error = _unigram_error(document, model)
    except RecursionError:
        return "tokenizer.json nesting exceeds the native validation limit"
    if error:
        return error
    return _added_tokens_error(document, model_type, len(model["vocab"]))
