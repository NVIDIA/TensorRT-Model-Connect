# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resident pinned upstream Lance image-understanding reference, with the Task's 448px input."""

from __future__ import annotations

import itertools
import json
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from families.lance.tests.benchmark import upstream_reference as reference

IMAGE_SIZE = 448


def image_values(image: Any) -> np.ndarray:
    """The public Task's fixed square image size, with the upstream CLIP normalization."""
    from PIL import Image

    pixels = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC),
                        dtype=np.float32) / 255.0
    values = (pixels - np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)) / np.asarray(
        [0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1))


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        if spec.mode != "eager" or spec.precision != "bf16":
            raise host.Error("Lance reference requires eager bf16")
        self.host, self.spec = host, spec
        self.requests, self.responses = queue.Queue(), queue.Queue()
        self.ready, self.closed = threading.Event(), threading.Event()
        self.lock, self.error = threading.Lock(), None
        self.temporary = tempfile.TemporaryDirectory(prefix="trtmc-lance-native-")
        self.root = Path(self.temporary.name)
        self.worker = threading.Thread(target=self._run, name="lance-native", daemon=True)
        self.worker.start()
        if not self.ready.wait(600):
            self.close()
            raise host.Error("Lance native model loading timed out")
        if self.error:
            self.close()
            raise host.Error(f"Lance native model loading failed: {self.error}")

    def _run(self) -> None:
        try:
            self._serve()
        except BaseException as error:
            self.error = error
            self.ready.set()
            self.responses.put(error)

    def _serve(self) -> None:
        checkout = Path(sys.prefix) / "trtmc-reference/Lance"
        arguments = reference.build_parser().parse_args([
            "--reference-repo", str(checkout), "--model", self.spec.model,
            "--image", str(self.root / "unused.png"), "--prompt", "", "--max-new-tokens", "16",
            "--warmup", "0", "--iterations", "1", "--resolution", "image_512res",
            "--height", "448", "--width", "448", "--output", str(self.root / "unused.json"),
            "--local-files-only", *(["--revision", self.spec.revision] if self.spec.revision else [])])
        model, vit, vae = reference._model_paths(arguments)
        reference._require_reference(checkout)
        sys.path.insert(0, str(checkout))
        reference._configure_upstream_vae(checkout, vae)
        upstream = reference._load_upstream(checkout)
        vit = reference._sdpa_vit_path(self.root, vit)
        dataset_file = self.root / "input.json"
        dataset_file.write_text(json.dumps(reference._dataset_payload(
            image=self.root / "unused.png", prompt="", instruction="", count=1)))
        original_loader, original_validate = upstream.DataLoader, upstream.validate_on_fixed_batch
        adapter = self

        class ResidentLoader:
            def __init__(self, dataset: Any, **kwargs: Any) -> None:
                self.dataset = dataset
                dataset.get_video_tensor_online = adapter._image_tensor
                adapter.ready.set()  # upstream has finished loading model, tokenizer, and dataset machinery

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return self

            def __next__(self):
                request = adapter.requests.get()
                if request is None:
                    raise StopIteration
                upstream.MAX_GENERATION_LENGTH = int(request.get("max_new_tokens", 16))
                self.dataset.data = [{"index": "0", "data": reference._dataset_payload(
                    image=Path(request["image_path"]), prompt=request["prompt"], instruction="", count=1)["0000"]}]
                # Keep the official collate and pin-memory path; only the source of the next sample changes.
                return next(iter(original_loader(self.dataset, batch_size=1, num_workers=0, pin_memory=True,
                                                 collate_fn=upstream.simple_custom_collate)))

        def validate(*args: Any, **kwargs: Any) -> Any:
            inference = kwargs["inference_args"]
            inference.prompt_data_dict = {}
            tokenizer = kwargs["tokenizer"]
            original_decode, token_ids = tokenizer.decode, []

            def decode(ids: Any, *arguments: Any, **options: Any) -> Any:
                # Official generated IDs, including EOS; do not retokenize normalized answers.
                token_ids.extend(int(value) for value in ids.tolist())
                return original_decode(ids, *arguments, **options)

            tokenizer.decode = decode
            try:
                value, elapsed = self.host.timed(lambda: original_validate(*args, **kwargs))
            finally:
                tokenizer.decode = original_decode
            text = upstream.normalize_understanding_answer(str(next(iter(inference.prompt_data_dict.values()), "")))
            self.responses.put(({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids),
                                 "input_image_height": IMAGE_SIZE,
                                 "input_image_width": IMAGE_SIZE}, elapsed))
            return value

        upstream.DataLoader, upstream.validate_on_fixed_batch = ResidentLoader, validate
        upstream.trange = lambda *args, **kwargs: itertools.count()
        argv = reference._upstream_argv(arguments, model, vit, dataset_file, self.root / "results")
        with reference._working_directory(checkout), reference._arguments(argv):
            upstream.main()

    def _image_tensor(self, media_url: str, vision_stream: str, worker_id: int = 0,
                      element_dtype: str = "image") -> Any:
        if vision_stream != "vit_video" or element_dtype != "image":
            raise self.host.Error("Lance native adapter supports image understanding only")
        import torch

        # Decoded by the serving host before its Task timer; resize/normalize stays inside the call.
        values = image_values(self.host.load_image(media_url))
        return torch.as_tensor(values).unsqueeze(1).repeat(1, 2, 1, 1)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        prompt = str(self.host.required(request, "prompt"))
        image = str(self.host.required(request, "image_path"))
        if int(request.get("max_new_tokens", 16)) <= 0 or float(request.get("temperature", 0)) != 0:
            raise self.host.Error("Lance reference requires a positive generation length and greedy decoding")
        with self.lock:
            if self.error or self.closed.is_set():
                raise self.host.Error(f"Lance native worker stopped: {self.error}")
            self.requests.put({**request, "prompt": prompt, "image_path": image})
            try:
                result = self.responses.get(timeout=1800)
            except queue.Empty:
                raise self.host.Error("Lance native request timed out") from None
            if isinstance(result, BaseException):
                raise self.host.Error(f"Lance native request failed: {result}") from result
            observation, elapsed = result
            return self.host.invocation(observation, elapsed)

    def close(self) -> None:
        self.closed.set()
        self.requests.put(None)
        self.worker.join(timeout=30)
        if not self.worker.is_alive():
            self.temporary.cleanup()
