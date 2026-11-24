# SPDX-License-Identifier: Apache-2.0
import json
import numpy as np
import tritonclient.http as http
import tritonclient.grpc as grpc
from tritonclient.utils import np_to_triton_dtype
from families.hstu.triton.native_client import NativeClient
from families.hstu.config import Config
from families.hstu.reference import example_inputs

from .common import root, runtime_root

admin = http.InferenceServerClient("localhost:8000")
(root / "server-metadata.json").write_text(json.dumps(admin.get_server_metadata(), indent=2) + "\n")
results = []
for candidates in (50, 64):
    for precision in ("fp32", "bf16"):
        variant = f"hstu{candidates}-{precision}"
        model = "native_" + variant.replace("-", "_")
        try:
            admin.load_model(model)
            config = admin.get_model_config(model)
            assert config["backend"] == "tensorrt" and config["platform"] == "model_connect"
            (root / "output" / variant / "native-served-config.json").write_text(
                json.dumps(config, indent=2)
            )
            with NativeClient(
                root / "output" / variant / "model.bundle", runtime_root
            ) as reference:
                for batch in (1, 4):
                    features = example_inputs(Config.load(root / "output" / variant), batch)
                    expected = reference.score(features)
                    for module, protocol, port in ((http, "http", 8000), (grpc, "grpc", 8001)):
                        with module.InferenceServerClient(f"localhost:{port}") as client:
                            tensors = []
                            for name, value in features.items():
                                value = np.ascontiguousarray(value)
                                tensor = module.InferInput(
                                    name, value.shape, np_to_triton_dtype(value.dtype)
                                )
                                tensor.set_data_from_numpy(value)
                                tensors.append(tensor)
                            response = client.infer(
                                model,
                                tensors,
                                outputs=[module.InferRequestedOutput(name) for name in expected],
                            )
                            for name, value in expected.items():
                                np.testing.assert_array_equal(response.as_numpy(name), value)
                        results.append(
                            dict(
                                variant=variant,
                                batch=batch,
                                protocol=protocol,
                                backend="tensorrt",
                                platform="model_connect",
                                bit_identical=True,
                            )
                        )
                        print(variant, batch, protocol, "PASS", flush=True)
                        (root / "native-triton-results.json").write_text(
                            json.dumps(results, indent=2) + "\n"
                        )
            (root / "output" / variant / "native-served-statistics.json").write_text(
                json.dumps(admin.get_inference_statistics(model_name=model), indent=2)
            )
        finally:
            admin.unload_model(model)
admin.close()
