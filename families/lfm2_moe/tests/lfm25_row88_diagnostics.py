# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded observations of the retained public full-card divergence, not a verdict."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from tools.e2e_evidence import record_evidence


_FIXTURE = Path(__file__).parent / "fixtures/lfm25-row88-public-prefix.json"
_ROWS = 89
_PROMPT_LENGTH = 16
_FINAL_LENGTH = 104


def _bits(tensor):
    """Make an immutable logical-order CPU copy without widening BF16 values."""
    import numpy as np
    import torch

    if tensor.dtype != torch.bfloat16:
        raise ValueError("state observation requires BF16")
    value = tensor.detach().to(device="cpu", copy=True).contiguous()
    result = value.view(torch.int16).numpy().view(np.uint16).copy()
    result.flags.writeable = False
    return result


def _descriptor(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
        "storage_offset": tensor.storage_offset(),
    }


class StateRows:
    """Retain actual used caches; append compression requires observed equality."""

    def __init__(self, config):
        import numpy as np

        self.types = list(config.layer_types)
        self.attention = [i for i, kind in enumerate(self.types) if kind == "full_attention"]
        self.conv = [i for i, kind in enumerate(self.types) if kind == "conv"]
        if (
            config.hidden_size != 2048
            or config.num_key_value_heads != 8
            or config.num_attention_heads != 32
            or config.conv_L_cache != 3
            or self.attention != [2, 6, 10, 14, 18, 21]
            or len(self.types) != 24
            or len(self.conv) != 18
        ):
            raise ValueError("state observation geometry is not the pinned MoE geometry")
        self.target = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        self.kv = np.zeros((6, 2, 8, _FINAL_LENGTH, 64), dtype=np.uint16)
        self.conv_values = np.zeros((18, 2048, _ROWS + 2), dtype=np.uint16)
        self.conv_previous = None
        self.conv_taps = 0
        self.conv_anomaly = None
        self.conv_anomaly_values = None
        self.conv_descriptors = None
        self.previous = None
        self.compacted_length = 0
        self.anomaly = None
        self.anomaly_values = None
        self.calls = []
        self.rows = []
        self.error = None

    def before(self, step, args, kwargs):
        if step >= _ROWS or self.error is not None:
            return
        try:
            ids = kwargs.get("input_ids", args[0] if args else None)
            positions = kwargs.get("cache_position")
            mask = kwargs.get("attention_mask")
            past = kwargs.get("past_key_values")
            expected_ids = (
                self.target["prompt_ids"]
                if step == 0
                else [self.target["reference_prefix"][step - 1]]
            )
            expected_positions = list(range(16)) if step == 0 else [15 + step]
            observed_ids = ids.detach().cpu().tolist()
            observed_positions = positions.detach().cpu().tolist()
            before_length = 0 if past is None else past.get_seq_length()
            if observed_ids != [expected_ids] or observed_positions != expected_positions:
                raise ValueError("original state observation history or positions differ")
            if before_length != (0 if step == 0 else 15 + step):
                raise ValueError("original state observation starts at the wrong cache length")
            self.calls.append(
                {
                    "row": step,
                    "input_ids": observed_ids,
                    "cache_position": observed_positions,
                    "cache_length_before": before_length,
                    "attention_mask_shape": None if mask is None else list(mask.shape),
                    "past_cache_type": type(past).__name__,
                }
            )
        except Exception as error:
            self.error = type(error).__name__

    def after(self, step, output):
        if step >= _ROWS or self.error is not None:
            return
        try:
            self._snapshot(step, output.past_key_values)
        except Exception as error:
            self.error = type(error).__name__

    def _snapshot(self, step, cache):
        import numpy as np

        length = _PROMPT_LENGTH + step
        if len(self.rows) != step or len(self.calls) != step + 1:
            raise ValueError("original state observation call pairing differs")
        if type(cache).__name__ != "Lfm2MoeHybridConvCache" or cache.get_seq_length() != length:
            raise ValueError("original state observation cache type or length differs")
        row = {
            "row": step,
            "position": length,
            "kv_prefix_unchanged": None,
            "conv_prefix_unchanged": None,
            "attention": [],
        }
        if self.anomaly is None:
            current = np.empty((6, 2, 8, length, 64), dtype=np.uint16)
            for index, layer in enumerate(self.attention):
                for kind, values in enumerate((cache.key_cache, cache.value_cache)):
                    tensor = values[layer]
                    if list(tensor.shape) != [1, 8, length, 64]:
                        raise ValueError("original active KV shape differs")
                    current[index, kind] = _bits(tensor)[0]
                    row["attention"].append(
                        {
                            "layer": layer,
                            "kind": "key" if kind == 0 else "value",
                            **_descriptor(tensor),
                        }
                    )
            self._append(step, current, row)
        current_conv = np.empty((18, 2048, 3), dtype=np.uint16)
        descriptors = []
        for index, layer in enumerate(self.conv):
            tensor = cache.conv_cache[layer]
            if list(tensor.shape) != [1, 2048, 3]:
                raise ValueError("original convolution state shape differs")
            current_conv[index] = _bits(tensor)[0]
            descriptors.append({"layer": layer, **_descriptor(tensor)})
        if self.conv_descriptors is None:
            self.conv_descriptors = descriptors
        elif descriptors != self.conv_descriptors:
            raise ValueError("original convolution state layout changed")
        self._append_conv(step, current_conv, row)
        row["conv_layout_rechecked"] = True
        self.rows.append(row)

    def _append(self, step, current, row):
        import numpy as np

        length = current.shape[-2]
        changed = None if self.previous is None else current[..., :-1, :] != self.previous
        if changed is not None and changed.any():
            coordinate = np.argwhere(changed)[0].tolist()
            self.anomaly = {
                "row": step,
                "position": length,
                "coordinate": coordinate,
                "layer": self.attention[coordinate[0]],
                "kind": "key" if coordinate[1] == 0 else "value",
                "differing_words": int(changed.sum()),
            }
            self.anomaly_values = current.copy()
            row["kv_prefix_unchanged"] = False
            return
        if self.previous is None:
            self.kv[..., :length, :] = current
        else:
            self.kv[..., length - 1, :] = current[..., -1, :]
        self.compacted_length = length
        self.previous = current.copy()
        row["kv_prefix_unchanged"] = True

    def _append_conv(self, step, current, row):
        import numpy as np

        if self.conv_anomaly is not None:
            return
        changed = (
            None if self.conv_previous is None else current[..., :2] != self.conv_previous[..., 1:]
        )
        if changed is not None and changed.any():
            coordinate = np.argwhere(changed)[0].tolist()
            self.conv_anomaly = {
                "row": step,
                "position": 16 + step,
                "coordinate": coordinate,
                "layer": self.conv[coordinate[0]],
                "differing_words": int(changed.sum()),
            }
            self.conv_anomaly_values = current.copy()
            row["conv_prefix_unchanged"] = False
            return
        if self.conv_previous is None:
            self.conv_values[..., :3] = current
            self.conv_taps = 3
        else:
            self.conv_values[..., self.conv_taps] = current[..., -1]
            self.conv_taps += 1
        self.conv_previous = current.copy()
        row["conv_prefix_unchanged"] = True

    def report(self):
        import numpy as np

        complete = len(self.rows) == _ROWS and self.error is None
        result = {
            "status": (
                "prefix_changed"
                if self.anomaly is not None or self.conv_anomaly is not None
                else "captured"
            )
            if complete
            else "unavailable",
            "error_type": self.error,
            "rows_captured": len(self.rows),
            "layer_types": self.types,
            "attention_layers": self.attention,
            "conv_layers": self.conv,
            "calls": self.calls,
            "rows": self.rows,
            "semantic_dtype": "bfloat16",
            "storage_dtype": "uint16",
            "kv_compacted_length": self.compacted_length,
            "prefix_anomaly": self.anomaly,
            "kv_reconstructible_through_length": self.compacted_length,
            "conv_prefix_anomaly": self.conv_anomaly,
            "conv_compacted_taps": self.conv_taps,
            "conv_reconstructible_through_row": self.conv_taps - 3 if self.conv_taps else None,
            "conv_descriptors": self.conv_descriptors,
            "conv_axis_order": ["conv_layer", "hidden", "history_tap"],
            "kv_history_bits": np.ascontiguousarray(self.kv[..., : self.compacted_length, :]),
            "conv_history_bits": np.ascontiguousarray(self.conv_values[..., : self.conv_taps]),
            "prior_artifact_anchor": "Requires external full reference IDs and row88 byte comparison",
            "scope": "Used cache tensors only; original conv placeholders at attention layers are excluded",
        }
        if self.anomaly_values is not None:
            result["prefix_anomaly_bits"] = self.anomaly_values
        if self.conv_anomaly_values is not None:
            result["conv_prefix_anomaly_bits"] = self.conv_anomaly_values
        return result


class Layer0Prefix:
    """Observe existing row1 values; the newest cache tap is never recomputed."""

    def __init__(self, owner):
        self.owner = owner
        self.handles = []
        self.values = {}
        self.descriptors = {}
        self.sources = {}
        self.counts = {}
        self.error = None
        self.hooks_removed = False

    def attach(self, model):
        import inspect

        try:
            layer = model.model.layers[0]
            modules = {
                "embedding": (model.model.embed_tokens, [1, 1, 2048]),
                "operator_norm": (layer.operator_norm, [1, 1, 2048]),
                "in_proj": (layer.conv.in_proj, [1, 1, 6144]),
            }
            methods = {name: module.forward for name, (module, _) in modules.items()}
            methods.update(conv=layer.conv.forward, slow_conv=layer.conv.slow_forward)
            for name, method in methods.items():
                source = inspect.getsource(method)
                if len(source.encode("utf-8")) > 8192:
                    raise ValueError("prefix callable source exceeds observation bound")
                code = method.__func__.__code__
                self.sources[name] = {
                    "module": method.__module__,
                    "qualname": method.__qualname__,
                    "filename": Path(code.co_filename).name,
                    "first_line": code.co_firstlineno,
                    "source_text": source,
                }
            for name, (module, shape) in modules.items():
                self.handles.append(module.register_forward_hook(self._output(name, shape)))
            self.handles.append(layer.conv.register_forward_hook(self._conv, with_kwargs=True))
        except Exception as error:
            self.error = type(error).__name__

    def _retain(self, name, tensor, shape):
        if list(tensor.shape) != shape or name in self.values:
            raise ValueError("prefix tensor shape or call count differs")
        self.values[name] = _bits(tensor)
        self.descriptors[name] = _descriptor(tensor)
        self.counts[name] = self.counts.get(name, 0) + 1

    def _output(self, name, shape):
        def observe(module, args, output):
            if self.owner.calls == 1 and self.error is None:
                try:
                    self._retain(name, output, shape)
                except Exception as error:
                    self.error = type(error).__name__
            return None

        return observe

    def _conv(self, module, args, kwargs, output):
        if self.owner.calls == 1 and self.error is None:
            try:
                cache = kwargs["past_key_values"]
                tensor = cache.conv_cache[0]
                if list(tensor.shape) != [1, 2048, 3]:
                    raise ValueError("prefix convolution cache shape differs")
                self._retain("newest_conv_tap", tensor[..., -1:], [1, 2048, 1])
            except Exception as error:
                self.error = type(error).__name__
        return None

    def finish(self):
        import numpy as np

        self.hooks_removed = True
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as error:
                self.hooks_removed = False
                self.error = type(error).__name__
        state = self.owner.state
        expected = {"embedding", "operator_norm", "in_proj", "newest_conv_tap"}
        history = (
            state is not None
            and len(state.calls) > 1
            and state.calls[1]["input_ids"] == [[state.target["reference_prefix"][0]]]
            and state.calls[1]["cache_position"] == [16]
            and state.calls[1]["cache_length_before"] == 16
        )
        endpoint = (
            state is not None
            and len(state.rows) > 1
            and state.conv_taps >= 4
            and "newest_conv_tap" in self.values
            and np.array_equal(self.values["newest_conv_tap"][0, :, 0], state.conv_values[0, :, 3])
        )
        complete = set(self.values) == expected and all(x == 1 for x in self.counts.values())
        return {
            "status": "capture_error"
            if self.error
            else "captured"
            if complete and history and endpoint
            else "unavailable",
            "row": 1,
            "layer": 0,
            "input_ids": state.calls[1]["input_ids"] if history else None,
            "cache_position": [16] if history else None,
            "history_matches": bool(history),
            "endpoint_matches_root_state": bool(endpoint),
            "error_type": self.error,
            "hook_removed": self.hooks_removed,
            "calls": self.counts,
            "descriptors": self.descriptors,
            "sources": self.sources,
            "tensors": self.values,
            "semantic_dtype": "bfloat16",
            "storage_dtype": "uint16",
            "scope": "Passive outputs from the existing original generation. Newest cache tap is copied after layer0 convolution, not recomputed. External full IDs/raw and retained endpoint joins are required before cross-run interpretation.",
            "source_scope": "Bound callable locations and inspected source text; no replacement or claim about lower-level kernels.",
        }


class Row88Capture:
    """A passive root-forward hook; no generation argument, output or RNG change."""

    def __init__(self):
        self.calls = 0
        self.row = None
        self.handle = None
        self.error = None
        self.report = {"status": "not_captured", "row": 88}
        self.prompt_ids = []
        self.state = None
        self.state_handle = None
        self.state_error = None
        self.state_hook_removed = False
        self.prefix = Layer0Prefix(self)

    def attach(self, model):
        try:
            self.handle = model.register_forward_hook(self._observe)
        except Exception as error:
            self.error = type(error).__name__
        try:
            self.state = StateRows(model.config)
            self.state_handle = model.register_forward_pre_hook(self._before, with_kwargs=True)
        except Exception as error:
            self.state_error = type(error).__name__
        self.prefix.attach(model)

    def _before(self, module, args, kwargs):
        if self.state is not None:
            self.state.before(self.calls, args, kwargs)

    def _observe(self, module, args, output):
        index = self.calls
        self.calls += 1
        if index == 88:
            try:
                self.row = output.logits[0, -1, :].detach().clone()
                self.report.update(
                    output_shape=list(output.logits.shape), dtype=str(self.row.dtype)
                )
            except Exception as error:
                self.error = type(error).__name__
        if self.state is not None:
            self.state.after(index, output)
        return None

    def finish(self, prompt_ids):
        self.prompt_ids = list(prompt_ids)
        try:
            if self.handle is not None:
                self.handle.remove()
            if self.error is not None:
                self.report.update(status="capture_error", error_type=self.error)
            elif self.row is not None:
                self.report.update(status="captured", logits=self.row.float().cpu().numpy())
        except Exception as error:
            self.report.update(status="capture_error", error_type=type(error).__name__)
        finally:
            self.row = None
            if self.state_handle is not None:
                try:
                    self.state_handle.remove()
                    self.state_hook_removed = True
                except Exception as error:
                    self.state_error = type(error).__name__
        self.report.update(
            forward_calls=self.calls,
            prompt_ids=self.prompt_ids,
            scope="Root forward logits before generation processors; hook returns None.",
        )
        try:
            prefix_report = self.prefix.finish()
        except Exception as error:
            prefix_report = {"status": "capture_error", "error_type": type(error).__name__}
        record_evidence("reference_row88", self.report)
        try:
            state_report = (
                self.state.report()
                if self.state is not None
                else {
                    "status": "unavailable",
                    "error_type": self.state_error,
                }
            )
            state_report["hook_removed"] = self.state_handle is None or self.state_hook_removed
            if self.state_error is not None:
                state_report.update(status="unavailable", error_type=self.state_error)
            record_evidence("reference_state_rows", state_report)
        except Exception as error:
            try:
                record_evidence(
                    "reference_state_rows",
                    {
                        "status": "capture_error",
                        "error_type": type(error).__name__,
                    },
                )
            except Exception:
                pass  # Do not replace the original generation or semantic verdict.
        try:
            record_evidence("reference_layer0_row1", prefix_report)
        except Exception:
            pass  # Prefix observation cannot replace the original verdict either.


def capture_for(manifest, case):
    if (
        manifest.get("hf_id") == "LiquidAI/LFM2.5-8B-A1B"
        and manifest.get("hf_revision") == "5dd22602c2e9f6a097b1de4c4efe0658b605015c"
        and case.get("name") == "lfm25-8b-a1b-model-card-greedy-full"
    ):
        return Row88Capture()
    return None


def collect_native(capture, bundle, prompt, actual_ids, reference_ids, tmp_path):
    """Record missing/failed observations, then let the unchanged semantic check run."""
    if capture is None:
        return
    report = {"status": "unavailable", "row": 88, "acceptance_changed": False}
    try:
        _collect_native(report, capture, bundle, prompt, actual_ids, reference_ids, tmp_path)
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__)
    record_evidence("paired_row88", report)


def _collect_native(report, capture, bundle, prompt, actual_ids, reference_ids, tmp_path):
    import numpy as np

    target = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    report.update(
        public_run=target["public_run"],
        expected_reference_id=target["reference_prefix"][88],
        expected_native_id=target["native_prefix"][88],
        reference_prefix=reference_ids[:89],
        actual_native_prefix=actual_ids[:89],
    )
    checks = {
        "reference_prefix": reference_ids[:89] == target["reference_prefix"],
        "native_prefix": actual_ids[:89] == target["native_prefix"],
        "shared_history": reference_ids[:88] == actual_ids[:88],
        "reference_prompt": capture.prompt_ids == target["prompt_ids"],
        "reference_capture": capture.report["status"] == "captured",
        "reference_call_count": capture.calls == len(reference_ids),
    }
    report["admission"] = checks
    if not all(checks.values()):
        report["status"] = "target_not_reproduced"
        return
    reference_logits = np.asarray(capture.report["logits"])
    checks["reference_raw_argmax"] = (
        reference_logits.shape == (128000,)
        and bool(np.isfinite(reference_logits).all())
        and int(reference_logits.argmax()) == target["reference_prefix"][88]
    )
    if not checks["reference_raw_argmax"]:
        report["status"] = "target_not_reproduced"
        return
    build = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    built = subprocess.run(
        ["cmake", "--build", str(build), "--target", "lfm2_moe_logits_probe", "--parallel", "2"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    report["probe_build_returncode"] = built.returncode
    build_log = tmp_path / "row88-probe-build.log"
    build_log.write_text(built.stdout + built.stderr)
    report["probe_build_log"] = build_log
    if built.returncode:
        report["status"] = "probe_build_failed"
        return
    probe = build / "families/lfm2_moe/lfm2_moe_logits_probe"
    encoded = subprocess.run(
        [str(probe), "--encode-prompt", str(bundle), prompt],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    report["prompt_probe_returncode"] = encoded.returncode
    prompt_log = tmp_path / "row88-prompt-probe.log"
    prompt_log.write_text(encoded.stderr)
    report["prompt_probe_log"] = prompt_log
    if encoded.returncode:
        report["status"] = "prompt_probe_failed"
        return
    native_prompt = json.loads(encoded.stdout)["prompt_ids"]
    report["native_prompt_ids"] = native_prompt
    report["prompt_scope"] = (
        "Reconstructed using production renderer/tokenizer helpers, not intercepted from the earlier CLI."
    )
    checks["native_prompt"] = native_prompt == capture.prompt_ids == target["prompt_ids"]
    if not checks["native_prompt"]:
        report["status"] = "target_not_reproduced"
        return
    prefill, row = tmp_path / "row88-native-prefill.f32", tmp_path / "row88-native.f32"
    requests = [
        {"reset": True, "tokens": native_prompt, "decode_phase": False, "output": str(prefill)},
        {"tokens": reference_ids[:88], "decode_phase": True, "output": str(row)},
    ]
    state_enabled = (
        capture.state is not None
        and capture.state.error is None
        and len(capture.state.rows) == _ROWS
        and capture.state_error is None
        and (capture.state_handle is None or capture.state_hook_removed)
    )
    state_directory = tmp_path / "row88-native-state"
    observed_row = tmp_path / "row88-native-observed.f32"
    if state_enabled:
        requests.extend(
            [
                {
                    "reset": True,
                    "tokens": native_prompt,
                    "decode_phase": False,
                    "output": str(tmp_path / "row88-native-observed-prefill.f32"),
                    "state_capture": str(state_directory),
                },
                {"tokens": reference_ids[:88], "decode_phase": True, "output": str(observed_row)},
            ]
        )
    completed = subprocess.run(
        [str(probe), str(bundle)],
        input="".join(json.dumps(item) + "\n" for item in requests),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    report["native_probe_returncode"] = completed.returncode
    native_log = tmp_path / "row88-native-probe.log"
    native_log.write_text(completed.stdout + completed.stderr)
    report["native_probe_log"] = native_log
    if completed.returncode and not state_enabled:
        report["status"] = "native_probe_failed"
        return
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    if completed.returncode and (len(responses) < 3 or not row.is_file()):
        report["status"] = "native_probe_failed"
        return
    checks["native_positions"] = (
        (len(responses) >= 3 if state_enabled else len(responses) == 3)
        and responses[0] == {"ready": True}
        and responses[1] == {"position": 16, "vocab_size": 128000}
        and responses[2] == {"position": 104, "vocab_size": 128000}
    )
    native_logits = np.fromfile(row, dtype=np.float32)
    report.update(
        native_logits=native_logits, shared_history_ids=native_prompt + reference_ids[:88]
    )
    checks["native_raw_argmax"] = (
        native_logits.shape == (128000,)
        and bool(np.isfinite(native_logits).all())
        and int(native_logits.argmax()) == target["native_prefix"][88]
    )
    if native_logits.shape == (128000,) and bool(np.isfinite(native_logits).all()):
        for name, values in (("reference", reference_logits), ("native", native_logits)):
            indices = np.argsort(-values, kind="stable")[:5]
            report[name + "_top_logits"] = [
                {"token_id": int(index), "value": float(values[index])} for index in indices
            ]
            report[name + "_top_margin"] = float(values[indices[0]]) - float(values[indices[1]])
    report["status"] = "admitted" if all(checks.values()) else "target_not_reproduced"
    if completed.returncode:
        report["status"] = (
            "baseline_captured_observer_failed" if all(checks.values()) else "native_probe_failed"
        )
    if state_enabled:
        _record_native_states(state_directory, observed_row, native_logits, responses, report)


def _native_bits(path, shape):
    import numpy as np

    expected = int(np.prod(shape)) * 2
    if expected > 8 * 1024 * 1024 or path.stat().st_size != expected:
        raise ValueError("native state payload size differs")
    return np.fromfile(path, dtype=np.uint16).reshape(shape)


def _native_history_matches(metadata):
    target = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    rows = metadata["rows"]
    if len(rows) != _ROWS:
        return False
    for index, row in enumerate(rows):
        expected = {
            "position": 16 + index,
            "write_index": 15 + index,
            "token": target["prompt_ids"][-1]
            if index == 0
            else target["reference_prefix"][index - 1],
            "phase": 0 if index == 0 else 1,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            return False
    for checks_name, anomaly_name, length_name, initial in (
        ("prefix_checks", "prefix_anomaly", "kv_compacted_length", 16),
        ("conv_prefix_checks", "conv_prefix_anomaly", "conv_compacted_taps", 3),
    ):
        checks, anomaly = metadata[checks_name], metadata.get(anomaly_name)
        if len(checks) != _ROWS or (checks[0] is not None and checks[0] is not True):
            return False
        if anomaly is None:
            if metadata[length_name] != initial + 88 or not all(
                item is True for item in checks[1:]
            ):
                return False
        else:
            row = anomaly["row"]
            if not (
                0 < row < _ROWS
                and anomaly["position"] == 16 + row
                and metadata[length_name] == initial + row - 1
                and all(item is True for item in checks[1:row])
                and checks[row] is False
                and all(item is None for item in checks[row + 1 :])
            ):
                return False
    return True


def _native_layout_matches(metadata):
    """Join actual context records; a producer's boolean is not layout evidence."""
    try:
        records = metadata["descriptors"]
        tensors = {item["name"]: item for item in records}
        if len(records) != 60 or len(tensors) != 60:
            return False
        capacity = tensors["cache_k_0"]["shape"][2]
        if not isinstance(capacity, int) or not 104 <= capacity <= 128000:
            return False
        expected, pairs = {}, []
        for index in range(6):
            for kind in ("k", "v"):
                source, output = f"cache_{kind}_{index}", f"present_{kind}_{index}"
                shape, strides = [1, 8, capacity, 64], [8 * capacity * 64, capacity * 64, 64, 1]
                expected[source], expected[output] = (
                    ("INPUT", shape, strides),
                    ("OUTPUT", shape, strides),
                )
                pairs.append((source, output, True))
        for index in range(18):
            source, output = f"conv_state_{index}", f"present_conv_{index}"
            expected[source], expected[output] = (
                ("INPUT", [2048, 3], [3, 1]),
                ("OUTPUT", [2048, 3], [3, 1]),
            )
            pairs.append((source, output, False))
        if set(tensors) != set(expected):
            return False
        for name, (mode, shape, strides) in expected.items():
            item = tensors[name]
            if (
                item["mode"] != mode
                or item["shape"] != shape
                or item["strides"] != strides
                or item["dtype"] != "BF16"
                or item["format"] != "LINEAR"
                or item["profile"] != 0
                or item["vectorized_dimension"] != -1
                or item["scalar_bytes"] != 2
                or (item["components_per_element"], item["bytes_per_component"])
                not in ((1, 2), (-1, -1))
                or not isinstance(item["address"], int)
                or item["address"] <= 0
                or item["address"] % 2
            ):
                return False
        if any((tensors[a]["address"] == tensors[b]["address"]) != alias for a, b, alias in pairs):
            return False
        rows = metadata["rows"]
        return (
            len(rows) == _ROWS
            and all(row["layout_rechecked"] is True for row in rows)
            and len({row["stream"] for row in rows}) == 1
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _record_native_states(directory, observed_row, baseline, responses, paired):
    """State observations have their own admission; baseline fields stay intact."""
    import numpy as np

    report = {"status": "unavailable", "acceptance_changed": False}
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        report["metadata"] = metadata
        attention = [2, 6, 10, 14, 18, 21]
        if metadata["attention_layers"] != attention or metadata["conv_layers"] != [
            i for i in range(24) if i not in attention
        ]:
            raise ValueError("native state global layer mapping differs")
        report["observed_positions"] = responses[3:] == [
            {"position": 16, "vocab_size": 128000},
            {"position": 104, "vocab_size": 128000},
        ]
        report["actual_history_and_prefix_metadata_valid"] = _native_history_matches(metadata)
        report["actual_layout_metadata_valid"] = _native_layout_matches(metadata)
        count, length = metadata["rows_captured"], metadata["kv_compacted_length"]
        taps = metadata["conv_compacted_taps"]
        if not (0 <= count <= _ROWS and 0 <= length <= _FINAL_LENGTH and 0 <= taps <= 91):
            raise ValueError("native state row range differs")
        kv_shape, conv_shape = [6, 2, 8, length, 64], [18, 2048, taps]
        if metadata["kv_history_shape"] != kv_shape or metadata["conv_shape"] != conv_shape:
            raise ValueError("native state payload shape differs")
        report["kv_history_bits"] = _native_bits(directory / "kv.u16", kv_shape)
        report["conv_history_bits"] = _native_bits(directory / "conv.u16", conv_shape)
        anomaly = metadata.get("prefix_anomaly")
        conv_anomaly = metadata.get("conv_prefix_anomaly")
        report["kv_reconstruction_complete"] = anomaly is None and length == _FINAL_LENGTH
        report["kv_reconstructible_through_length"] = length
        report["conv_reconstruction_complete"] = conv_anomaly is None and taps == 91
        report["conv_reconstructible_through_row"] = taps - 3 if taps else None
        if anomaly is not None:
            length = anomaly["position"]
            if not 16 < length <= _FINAL_LENGTH:
                raise ValueError("native prefix anomaly range differs")
            report["prefix_anomaly_bits"] = _native_bits(
                directory / "kv-prefix-anomaly.u16", [6, 2, 8, length, 64]
            )
        if conv_anomaly is not None:
            if (
                not 0 < conv_anomaly["row"] < 89
                or conv_anomaly["position"] != 16 + conv_anomaly["row"]
            ):
                raise ValueError("native convolution anomaly range differs")
            report["conv_prefix_anomaly_bits"] = _native_bits(
                directory / "conv-shift-anomaly.u16", [18, 2048, 3]
            )
        if metadata.get("present_anomaly") is not None:
            report["present_anomaly_bits"] = _native_bits(
                directory / "conv-present-anomaly.u16", [18, 2048, 3]
            )
        row_equal = False
        try:
            actual = np.fromfile(observed_row, dtype=np.float32)
            if actual.shape != (128000,) or not bool(np.isfinite(actual).all()):
                raise ValueError("observed row88 logits are invalid")
            report["observed_row88_logits"] = actual
            report["observed_row88_status"] = "available"
            row_equal = baseline.shape == actual.shape and bool(
                np.array_equal(actual.view(np.uint32), baseline.view(np.uint32))
            )
        except (OSError, ValueError) as error:
            report["observed_row88_status"] = "unavailable"
            report["observed_row88_error_type"] = type(error).__name__
        report["row88_full_words_identical_to_unobserved_same_module"] = row_equal
        admitted = (
            paired["status"] == "admitted"
            and row_equal
            and report["observed_positions"]
            and report["actual_history_and_prefix_metadata_valid"]
            and report["actual_layout_metadata_valid"]
            and count == _ROWS
            and metadata["positions"] == list(range(16, 105))
            and metadata["layout_admitted"] is True
            and metadata["stream_admitted"] is True
            and metadata["present_equal"] is True
            and metadata.get("error_type") is None
        )
        report["status"] = (
            (
                "admitted_state_anomaly"
                if anomaly is not None or conv_anomaly is not None
                else "admitted"
            )
            if admitted
            else "not_admitted"
        )
        report["scope"] = (
            "Same-module unobserved/observed row88 replay; state equality is diagnostic. "
            "Compact KV is reconstructible only through its recorded invariant boundary."
        )
    except Exception as error:
        report.update(status="capture_error", error_type=type(error).__name__)
    try:
        record_evidence("native_state_rows", report)
    except Exception:
        pass
