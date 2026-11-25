# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton Python transport for the standalone native structured-ranking task."""

import ctypes
import json
import threading
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        parameters = {
            key: value["string_value"] for key, value in config.get("parameters", {}).items()
        }
        runtime_root = parameters["runtime_root"]
        self.library = ctypes.CDLL(str(Path(runtime_root) / "libhstu_research_triton.so"))
        self.library.hstu_research_create.argtypes = [ctypes.c_char_p]
        self.library.hstu_research_create.restype = ctypes.c_void_p
        self.library.hstu_research_score.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.library.hstu_research_score.restype = ctypes.c_char_p
        self.library.hstu_research_history.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.library.hstu_research_history.restype = ctypes.c_char_p
        self.library.hstu_research_cache_stats.argtypes = [ctypes.c_void_p]
        self.library.hstu_research_cache_stats.restype = ctypes.c_char_p
        self.library.hstu_research_last_error.restype = ctypes.c_char_p
        self.library.hstu_research_destroy.argtypes = [ctypes.c_void_p]
        options = dict(
            bundle=parameters["bundle"],
            runtime_root=runtime_root,
            device=int(args["model_instance_device_id"]),
        )
        for key, convert in [("kv_cache_memory_ratio", float), ("kv_cache_size_bytes", int)]:
            if key in parameters:
                options[key] = convert(parameters[key])
                if options[key] <= 0:
                    raise ValueError(f"{key} must be positive when specified")
        if parameters.get("kv_cache_hydration_enabled", "false") not in ("true", "false"):
            raise ValueError("kv_cache_hydration_enabled must be true or false")
        self.hydration = parameters.get("kv_cache_hydration_enabled", "false") == "true"
        if self.hydration:
            options.update(
                kv_cache_hydration_enabled=True,
                kv_cache_max_entries=int(parameters.get("kv_cache_max_entries", 32)),
                kv_cache_ttl_seconds=float(parameters.get("kv_cache_ttl_seconds", 300)),
                kv_cache_lease_seconds=float(parameters.get("kv_cache_lease_seconds", 30)),
            )
            self.history_names = json.loads(parameters["history_input_names"])
            self.request_names = json.loads(parameters["request_input_names"])
        elif any(
            key in parameters
            for key in ("kv_cache_max_entries", "kv_cache_ttl_seconds", "kv_cache_lease_seconds")
        ):
            raise ValueError("history cache settings require kv_cache_hydration_enabled")
        self.handle = self.library.hstu_research_create(
            json.dumps(options, allow_nan=False).encode()
        )
        if not self.handle:
            raise RuntimeError(self.library.hstu_research_last_error().decode())
        self.input_names = [
            spec["name"]
            for spec in config["input"]
            if spec["name"] not in ("CACHE_CONTROL", "CACHE_SNAPSHOT")
        ]
        self.output_names = [
            spec["name"]
            for spec in config["output"]
            if spec["name"] not in ("CACHE_STATUS", "CACHE_SNAPSHOT")
        ]
        self.families, self.metrics = {}, {}
        self.metric_values, self.metric_lock = {}, threading.Lock()
        labels = dict(
            model=args["model_name"],
            version=args["model_version"],
            device=args["model_instance_device_id"],
            instance=args["model_instance_name"],
        )
        initial = json.loads(self.library.hstu_research_cache_stats(self.handle))
        for key in initial:
            family = pb.MetricFamily(
                name=f"hstu_kv_cache_{key}",
                description=f"Native KV cache {key}",
                kind=pb.MetricFamily.COUNTER if key.endswith("_total") else pb.MetricFamily.GAUGE,
            )
            self.families[key] = family
            self.metrics[key] = family.Metric(labels=labels)
        self.stop = threading.Event()
        self.publish()
        # Snapshot reads only a family mutex; scrapes never synchronize CUDA.
        self.publisher = threading.Thread(target=self.poll, daemon=True)
        self.publisher.start()

    def publish(self):
        with self.metric_lock:
            value = self.library.hstu_research_cache_stats(self.handle)
            if not value:
                raise RuntimeError(self.library.hstu_research_last_error().decode())
            for key, value in json.loads(value).items():
                if key.endswith("_total"):
                    self.metrics[key].increment(float(value) - self.metric_values.get(key, 0))
                    self.metric_values[key] = float(value)
                else:
                    self.metrics[key].set(float(value))

    def arrays(self, request, names):
        arrays = []
        for name in names:
            tensor = pb.get_input_tensor_by_name(request, name)
            if tensor is None:
                raise ValueError(f"missing input for operation: {name}")
            arrays.append(np.ascontiguousarray(tensor.as_numpy()))
        specs = [
            dict(
                name=name,
                dtype=str(value.dtype),
                shape=list(value.shape),
                bytes=value.nbytes,
                pointer=value.ctypes.data,
            )
            for name, value in zip(names, arrays)
        ]
        return arrays, specs

    def history_execute(self, request):
        control = pb.get_input_tensor_by_name(request, "CACHE_CONTROL").as_numpy().reshape(-1)
        if control.size != 1:
            raise ValueError("CACHE_CONTROL must contain one JSON string")
        command = json.loads(control[0])
        operation = command["operation"]
        arguments = dict(operation=operation)
        arrays = []
        if operation == "export":
            arrays, specs = self.arrays(request, self.history_names)
            arguments.update(inputs=specs, revisions=command["revisions"])
        elif operation == "restore":
            snapshot = pb.get_input_tensor_by_name(request, "CACHE_SNAPSHOT")
            if snapshot is None:
                raise ValueError("restore requires CACHE_SNAPSHOT")
            arrays = [np.ascontiguousarray(snapshot.as_numpy())]
            arguments.update(pointer=arrays[0].ctypes.data, bytes=arrays[0].nbytes)
        elif operation == "score":
            arrays, specs = self.arrays(request, self.request_names)
            arguments.update(
                inputs=specs, handles=command["handles"], revisions=command["revisions"]
            )
        elif operation == "info":
            arguments["handle"] = command["handle"]
        elif operation == "release":
            arguments["handles"] = command["handles"]
        else:
            raise ValueError(f"unknown history operation: {operation}")
        reply = self.library.hstu_research_history(self.handle, json.dumps(arguments).encode())
        if not reply:
            raise RuntimeError(self.library.hstu_research_last_error().decode())
        reply = json.loads(reply)
        outputs = []
        if operation == "export":
            data = (ctypes.c_uint8 * reply["bytes"]).from_address(reply["pointer"])
            outputs.append(pb.Tensor("CACHE_SNAPSHOT", np.frombuffer(data, dtype=np.uint8).copy()))
            reply = dict(exported=True, bytes=reply["bytes"])
        elif operation == "score":
            for spec in reply["outputs"]:
                data = (ctypes.c_uint8 * spec["bytes"]).from_address(spec["pointer"])
                outputs.append(
                    pb.Tensor(
                        spec["name"],
                        np.frombuffer(data, dtype=spec["dtype"]).reshape(spec["shape"]).copy(),
                    )
                )
            reply = dict(scored=True)
        outputs.append(
            pb.Tensor("CACHE_STATUS", np.array([json.dumps(reply).encode()], dtype=object))
        )
        return outputs

    def poll(self):
        while not self.stop.wait(0.01):
            try:
                self.publish()
            except Exception as error:
                pb.Logger.log_error(f"HSTU cache metrics update failed: {error}")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                if request.is_cancelled():
                    raise RuntimeError("request cancelled before native execution")
                if self.hydration:
                    outputs = self.history_execute(request)
                    self.publish()
                    responses.append(pb.InferenceResponse(output_tensors=outputs))
                    continue
                arrays, specs = self.arrays(request, self.input_names)
                result = self.library.hstu_research_score(self.handle, json.dumps(specs).encode())
                if not result:
                    raise RuntimeError(self.library.hstu_research_last_error().decode())
                outputs = []
                for spec in json.loads(result):
                    data = (ctypes.c_uint8 * spec["bytes"]).from_address(spec["pointer"])
                    value = np.frombuffer(data, dtype=spec["dtype"]).reshape(spec["shape"]).copy()
                    outputs.append(pb.Tensor(spec["name"], value))
                self.publish()
                responses.append(pb.InferenceResponse(output_tensors=outputs))
            except Exception as error:
                self.publish()
                responses.append(pb.InferenceResponse(error=pb.TritonError(str(error))))
        return responses

    def finalize(self):
        self.stop.set()
        self.publisher.join()
        self.metrics.clear()
        self.families.clear()
        self.library.hstu_research_destroy(self.handle)
        self.handle = None
