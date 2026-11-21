# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded observations of actual normalization operations; never an oracle."""

import importlib.metadata
import inspect
import json
import sys

import numpy as np


def _runs(words, limit=32):
    """Retain positions, rather than an unordered histogram of represented bits."""
    words = np.asarray(words).reshape(-1)
    result = []
    start = 0
    for end in range(1, words.size + 1):
        if end == words.size or words[end] != words[start]:
            if len(result) == limit:
                return None
            result.append({"start": start, "length": end - start, "bits": int(words[start])})
            start = end
    return result


def tensor_observation(value):
    """Describe original layout and a bounded, explicitly qualified CPU payload."""
    result = {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype),
        "device": value.device.type,
        "logical_order": "row-major indices of the recorded shape",
    }
    if value.device.type != "cpu":
        return {**result, "payload": "unavailable: non-CPU tensor"}
    array = value.detach().numpy()
    result["elements"] = array.size
    if array.dtype.kind == "f":
        result["finite_elements"] = int(np.isfinite(array).sum())
    if array.dtype == np.float32:
        words = array.view(np.uint32)
        if array.size <= 256:
            return {**result, "payload": "all_fp32_words", "words": words.reshape(-1).tolist()}
        if array.ndim == 3 and array.shape[0] == 1:
            same = bool(np.all(words == words[:, :1, :]))
            runs = _runs(words[0, 0])
            result.update(
                all_channels_bit_identical=same,
                first_channel_index=[0, 0],
                first_channel_rle=runs,
                logical_axes=["batch", "feature", "time"],
            )
            if same and runs is not None:
                return {**result, "payload": "lossless_first_channel_rle_repeated"}
        runs = _runs(words)
        if runs is not None:
            return {**result, "payload": "lossless_logical_flat_rle", "rle": runs}
    elif array.size <= 256 and array.dtype.kind in "biu":
        return {**result, "payload": "all_integer_values", "values": array.reshape(-1).tolist()}
    return {**result, "payload": "bounded_unavailable: not reconstructable from retained data"}


def _source_context(function):
    """Optional inspected source context, not executed-bytecode or kernel proof."""
    try:
        source = inspect.getsource(function)
        if len(source.encode("utf-8")) > 8192:
            return {"unavailable": "size bound"}
        return {"inspected_source_text": source}
    except Exception as error:
        return {"unavailable": type(error).__name__}


class NormalizationObservation:
    """Observe one exact normalize_batch frame while its original operators run."""

    def __init__(self, torch, normalize):
        self.torch = torch
        self.normalize = normalize
        self.frame = None
        self.previous_profile = None
        self.mode = None
        self.profile_installed = False
        self.mode_installed = False
        self.data = {
            "callable": f"{normalize.__module__}.{normalize.__qualname__}",
            "reference_source_context": _source_context(normalize),
            "torch_version": str(torch.__version__),
            "torch_revision": torch.version.git_version,
            "normalization_calls": 0,
            "normalization_returns": 0,
            "operations": [],
            "errors": [],
            "native_intermediates_observed": False,
        }
        try:
            self.data["liquid_audio_version"] = importlib.metadata.version("liquid-audio")
        except importlib.metadata.PackageNotFoundError:
            self.data["liquid_audio_version"] = "unavailable"

    def _snapshot(self, value):
        if isinstance(value, self.torch.Tensor):
            return tensor_observation(value)
        if isinstance(value, (list, tuple)):
            return [self._snapshot(item) for item in value[:8]]
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        return {"unavailable_type": type(value).__name__}

    def _observe(self, action):
        try:
            action()
        except Exception as error:
            if len(self.data["errors"]) < 8:
                self.data["errors"].append(type(error).__name__)

    def _profile(self, frame, event, value):
        if self.previous_profile is not None:
            self.previous_profile(frame, event, value)
        if frame.f_code is not self.normalize.__code__:
            return
        if event == "call":
            self.data["normalization_calls"] += 1
            self._observe(
                lambda: self.data.update(
                    normalization_input=self._snapshot(frame.f_locals["x"]),
                    sequence_length=self._snapshot(frame.f_locals["seq_len"]),
                )
            )
            self.frame = frame
        elif event == "return" and frame is self.frame:
            # Disable observation before snapshot operations made by this hook.
            self.frame = None
            self.data["normalization_returns"] += 1
            names = ("x_mean_numerator", "x_mean_denominator", "x_mean", "x_std")
            self._observe(
                lambda: self.data.update(
                    return_locals={name: self._snapshot(frame.f_locals[name]) for name in names},
                    returned=self._snapshot(value),
                    x_std_scope="Actual return local after NaN masking and epsilon addition",
                )
            )

    def __enter__(self):
        self.previous_profile = sys.getprofile()
        if self.previous_profile is not None and not callable(self.previous_profile):
            self.data["observation_unavailable"] = "active profiler is not callable"
            return self
        from torch.utils._python_dispatch import TorchDispatchMode

        owner = self

        class ObserveOperations(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                active = owner.frame is not None
                record = None
                if active:
                    if len(owner.data["operations"]) >= 32:
                        owner.data["operations_truncated"] = True
                    else:
                        record = {"ordinal": len(owner.data["operations"]), "operator": str(func)}
                        owner._observe(
                            lambda: record.update(
                                args=owner._snapshot(args),
                                kwargs={k: owner._snapshot(v) for k, v in kwargs.items()},
                            )
                        )
                # The original dispatch callable receives the original objects once.
                result = func(*args, **kwargs)
                if record is not None:
                    owner._observe(lambda: record.update(output=owner._snapshot(result)))
                    owner.data["operations"].append(record)
                    if str(func) == "aten.sqrt.default":
                        owner._observe(
                            lambda: owner.data.update(variance_before_sqrt=owner._snapshot(args[0]))
                        )
                return result

        self.mode = ObserveOperations()
        self.mode.__enter__()
        self.mode_installed = True
        try:
            sys.setprofile(self._profile)
            self.profile_installed = True
        except BaseException:
            self.mode.__exit__(*sys.exc_info())
            self.mode_installed = False
            raise
        return self

    def __exit__(self, *error):
        try:
            if self.profile_installed:
                sys.setprofile(self.previous_profile)
                self.profile_installed = False
            self.frame = None
        finally:
            if self.mode_installed:
                self.mode.__exit__(*error)
                self.mode_installed = False
        return False

    def report_failure(self, actual, expected, *, sample_count, processor_config):
        """Print observations only; preserve the caller's original assertion failure."""
        try:
            self._observe(
                lambda: self.data.update(
                    native_final_features=self._snapshot(actual),
                    original_final_features=self._snapshot(expected),
                )
            )
            report = {
                **self.data,
                "scope": "Same-call passive original normalization trace; no native reduction or cause claim",
                "input": {
                    "kind": "positive-zero float32 silence",
                    "samples": sample_count,
                    "sample_rate": 16000,
                },
                "final_feature_axes": ["time", "feature"],
                "processor_config": processor_config,
            }
            # Do not emit host paths, process environment, or unrelated values.
            encoded = json.dumps(report, allow_nan=False)
            if len(encoded) > 512 * 1024:
                encoded = json.dumps({"observation_unavailable": "bounded report size exceeded"})
            print("LFM2_AUDIO_NORMALIZATION_DIAGNOSTIC=" + encoded)
        except Exception as error:
            # Observational serialization must not replace assert_close's error.
            try:
                print(
                    "LFM2_AUDIO_NORMALIZATION_DIAGNOSTIC="
                    + json.dumps({"observation_unavailable": type(error).__name__})
                )
            except Exception:
                pass
