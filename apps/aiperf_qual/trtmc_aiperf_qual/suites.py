# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build fixed-sample suites from AIPerf benchmarks, our manifests, ETTh1 windows, or catalog testcases."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ConfigError, Environment, require


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def request_sha(request: Mapping[str, Any]) -> str:
    return sha256_text(canonical(request))


@dataclass(frozen=True)
class Suite:
    name: str
    key: str
    samples: list[dict[str, Any]]
    manifest: dict[str, Any]

    def write_inputs(self, path: Path) -> Path:
        """AIPerf single_turn file: each turn text is the JSON operation request."""
        with open(path, "w") as handle:
            for sample in self.samples:
                handle.write(json.dumps({"text": json.dumps({"request": sample["request"]})}) + "\n")
        return path

    def write_with_goldens(self, path: Path, goldens: Mapping[str, Any],
                           params: Mapping[str, Any] | None = None) -> Path:
        with open(path, "w") as handle:
            for sample in self.samples:
                handle.write(json.dumps({**sample, "golden": goldens[sample["request_sha"]],
                                         "params": dict(params or {})}) + "\n")
        return path


def with_latent_seeds(suite: Suite, base: int = 1000) -> Suite:
    """Each sample with its own ``latent_seed`` (base + index): trtmc-perf-serve then gives TRTMC and
    the native model the same initial diffusion noise (latent replay)."""
    samples = []
    for index, sample in enumerate(suite.samples):
        request = {**sample["request"], "latent_seed": base + index}
        samples.append({**sample, "request": request, "request_sha": request_sha(request)})
    key = sha256_text(canonical({"suite": suite.key, "latent_seed_base": base}))
    return Suite(suite.name, key, samples, {**suite.manifest, "key": key, "latent_seed_base": base})


def limit_suite(suite: Suite, count: int) -> Suite:
    """The first ``count`` samples (a new suite key, so goldens stay separate)."""
    if len(suite.samples) <= count:
        return suite
    samples = suite.samples[:count]
    key = sha256_text(canonical({"suite": suite.key, "limited_to": count}))
    return Suite(suite.name, key, samples, {**suite.manifest, "key": key, "samples": count, "limited_to": count})


def select(records: Sequence[dict[str, Any]], selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deterministic selection: first N, every k-th (stride to N), class-balanced (``stratified`` over a
    field), or explicit indices; optionally per task."""
    if selection.get("per_task"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(record.get("task", ""), []).append(record)
        inner = {key: value for key, value in selection.items() if key != "per_task"}
        return [item for task in groups for item in select(groups[task], inner)]
    method = require(selection, "method", "selection")
    if method == "first":
        return list(records[: int(require(selection, "count", "selection"))])
    if method == "stride":
        count = int(require(selection, "count", "selection"))
        step = max(len(records) // count, 1)
        return list(records[::step][:count])
    if method == "stratified":  # round-robin across the values of a field, in record order
        count, field = int(require(selection, "count", "selection")), require(selection, "field", "selection")
        groups = {}
        for position, record in enumerate(records):
            groups.setdefault(json.dumps(record.get(field), sort_keys=True), []).append((position, record))
        chosen = [group[depth] for depth in range(max(map(len, groups.values()), default=0))
                  for group in groups.values() if depth < len(group)][:count]
        return [record for _, record in sorted(chosen, key=lambda pair: pair[0])]
    if method == "indices":
        indices = require(selection, "indices", "selection")
        if any(index >= len(records) for index in indices):
            raise ConfigError(f"selection index out of range for {len(records)} records")
        return [records[index] for index in indices]
    raise ConfigError(f"unknown selection method {method!r}")


def _mmlu_records(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Zero-shot MMLU questions in lighteval's prompt format from a pinned dataset revision, read with
    the public ``datasets`` API; the answer letter is the label."""
    import datasets

    revision = require(source, "revision", "source")
    records = []
    for subject in require(source, "subjects", "source"):
        table = datasets.load_dataset(source.get("dataset", "lighteval/mmlu"), subject, split="test", revision=revision)
        for index in range(min(int(source.get("per_subject", 1)), len(table))):
            row = table[index]
            choices = "".join(f"\n{letter}. {text}" for letter, text in zip("ABCD", row["choices"]))
            prompt = (f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}."
                      f"\n\n{row['question'].strip()}{choices}\nAnswer:")
            answer = row["answer"]
            records.append({"id": f"{subject}/{index}", "task": subject, "prompt": prompt,
                            "label": "ABCD"[answer] if isinstance(answer, int) else str(answer)})
    return records


def _aiperf_public_records(source: Mapping[str, Any], selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Selected rows of an AIPerf public dataset, as registered (and pinned) in the plugin registry."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    name = require(source, "dataset", "source")
    loader = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, name)
    meta = plugins.get_public_dataset_loader_metadata(name)
    revision = getattr(loader, "hf_revision", None)
    if not revision:
        raise ConfigError(f"public dataset {name!r} is not pinned; register a pinned loader in trtmc-aiperf-plugins")
    return _hf_rows(name, meta, revision, source, selection)


def _hf_dataset_records(source: Mapping[str, Any], selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Selected rows of a Hugging Face dataset at a pinned revision (``hf_dataset``): ``dataset``,
    ``subset``, ``split``, ``revision``, and the ``prompt_column`` / ``image_column`` /
    ``audio_column`` / ``label_column`` it reads; ``columns`` copies further fields."""
    from types import SimpleNamespace

    meta = SimpleNamespace(hf_dataset_name=require(source, "dataset", "source"), hf_subset=source.get("subset"),
                           hf_split=require(source, "split", "source"), streaming=bool(source.get("streaming")),
                           data_files=source.get("data_files"),
                           **{f"{role}_column": source.get(f"{role}_column") for role in ("prompt", "image", "audio")})
    return _hf_rows(meta.hf_dataset_name, meta, require(source, "revision", "source"), source, selection)


VBENCH_RELATIONS = {"on the left of": "left of", "on the right of": "right of", "on the top of": "above",
                    "on the bottom of": "below"}


def vbench_object_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """VBench's object dimensions (object class, multiple objects, color, spatial relationship) as
    GenEval-style requirements of each prompt."""
    records = []
    for row in rows:
        info, prompt = row.get("auxiliary_info") or {}, row["prompt_en"]
        for dimension in row["dimension"]:
            if dimension == "object_class":
                include = [{"class": info[dimension]["object"], "count": 1}]
            elif dimension == "multiple_objects":
                include = [{"class": name.strip(), "count": 1} for name in info[dimension]["object"].split(" and ")]
            elif dimension == "color":
                color = info[dimension]["color"]
                name = prompt.split(f" {color} ", 1)[-1].split(",")[0].strip()
                include = [{"class": name, "count": 1, "color": color}]
            elif dimension == "spatial_relationship":
                relation = info[dimension]["spatial_relationship"]
                include = [{"class": relation["object_b"], "count": 1},
                           {"class": relation["object_a"], "count": 1,
                            "position": [VBENCH_RELATIONS[relation["relationship"]], 0]}]
            else:
                continue
            records.append({"id": str(len(records)), "prompt": prompt, "task": dimension,
                            "label": {"tag": dimension, "include": include, "prompt": prompt}})
    return records


def _url_jsonl_records(source: Mapping[str, Any], environment: Environment) -> list[dict[str, Any]]:
    """Records of a JSONL (or JSON list) file at ``url``, verified against ``sha256`` and cached below the
    datasets cache (``url_jsonl``); ``label_record`` makes the whole record the gold label, and
    ``transform: vbench_objects`` turns VBench's prompt list into GenEval-style requirements."""
    import urllib.request

    digest = require(source, "sha256", "source")
    path = Path(environment["hf_datasets_cache"]) / "downloads" / f"{digest}.jsonl"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(require(source, "url", "source"), timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ConfigError(f"{source['url']} does not match sha256 {digest}")
        path.write_bytes(data)
    text = path.read_text()
    records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines()
                                                                      if line.strip()]
    if source.get("transform") == "vbench_objects":
        return vbench_object_records(records)
    return [{**record, "id": str(index), **({"label": dict(record)} if source.get("label_record") else {}),
             **({"task": record.get(source["task_field"])} if source.get("task_field") else {})}
            for index, record in enumerate(records)]


def _image_archive_records(source: Mapping[str, Any], environment: Environment) -> list[dict[str, Any]]:
    """Images of a class-per-folder archive in a Hugging Face dataset repository (``image_archive``):
    ``repo``, ``revision``, ``filename``; the folder name is the integer class label. The archive is
    extracted once below the environment's datasets cache."""
    import tarfile

    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(require(source, "repo", "source"), require(source, "filename", "source"),
                                   repo_type="dataset", revision=require(source, "revision", "source")))
    root = Path(environment["hf_datasets_cache"]) / "archives" / f"{source['revision']}-{archive.name}"
    done = root / ".extracted"
    if not done.is_file():
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(root, filter="data")
        done.write_text(source["filename"])
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in (".jpeg", ".jpg", ".png") and path.parent.name.isdigit():
            records.append({"id": f"{path.parent.name}/{path.name}", "image": str(path), "label": int(path.parent.name)})
    return records


def polygon_mask(polygon: Any, size: Sequence[int]) -> Any:
    """A COCO polygon (flat [x, y, ...], or a list of them) rasterized as a boolean [height, width] mask."""
    import numpy as np
    from PIL import Image, ImageDraw

    parts = polygon if polygon and isinstance(polygon[0], (list, tuple)) else [polygon]
    canvas = Image.new("L", (int(size[0]), int(size[1])), 0)
    draw = ImageDraw.Draw(canvas)
    for part in parts:
        if len(part) >= 6:
            draw.polygon([(float(part[i]), float(part[i + 1])) for i in range(0, len(part) - 1, 2)], fill=1)
    return np.asarray(canvas, dtype=bool)


def _interior_point(polygon: Any, size: Sequence[int]) -> tuple[float, float]:
    """The mask pixel nearest the mask's centroid, normalized to [0, 1] (a point prompt inside the object)."""
    import numpy as np

    mask = polygon_mask(polygon, size)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return 0.5, 0.5
    nearest = int(np.argmin((rows - rows.mean()) ** 2 + (columns - columns.mean()) ** 2))
    return (float(columns[nearest]) + 0.5) / size[0], (float(rows[nearest]) + 0.5) / size[1]


def _hf_rows(name: str, meta: Any, revision: str, source: Mapping[str, Any],
             selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only selected rows are decoded. Audio becomes WAV (the TRTMC worker reads WAV); images keep
    their original encoded bytes."""
    import io

    import datasets
    import soundfile
    # Streaming datasets (for example LibriSpeech, whose config spans ~30 GB of train splits) are read
    # from the requested split only; media stay encoded until a row is selected.
    # ``data_files`` limits the download to the split's shards (for example ImageNet's validation parquet).
    # The dataset card lists every split, so a partial download skips the split verification.
    files = ({"data_files": meta.data_files, "verification_mode": "no_checks"}
             if getattr(meta, "data_files", None) else {})
    table = datasets.load_dataset(meta.hf_dataset_name, meta.hf_subset, split=meta.hf_split, revision=revision,
                                  streaming=bool(meta.streaming), **files)
    roles = {role: getattr(meta, f"{role}_column") for role in ("prompt", "image", "audio")
             if getattr(meta, f"{role}_column", None)}
    for role in ("image", "audio"):
        if role in roles:
            feature = datasets.Image if role == "image" else datasets.Audio
            table = table.cast_column(roles[role], feature(decode=False))
    label_column = source.get("label_column")
    if source.get("label_image"):  # an annotation image (e.g. a class map), kept encoded until scored
        table = table.cast_column(label_column, datasets.Image(decode=False))
    if meta.streaming:
        table = list(table)
    candidates = range(len(table))
    if source.get("where"):  # rows whose columns equal the given values (read before any media is decoded)
        columns = {name: (table[name] if not meta.streaming else [row[name] for row in table]) for name in source["where"]}
        candidates = [index for index in candidates
                      if all(columns[name][index] == value for name, value in source["where"].items())]
    # A stratified selection reads its field (the label column) before any media are decoded.
    field = selection.get("field") if selection.get("method") == "stratified" else None
    column = (label_column if field == "label" else field) if field else None
    values = (table[column] if not meta.streaming else [row[column] for row in table]) if column else None
    chosen = select([{"id": f"{name}/{index}", "row": index, **({field: values[index]} if column else {})}
                     for index in candidates], selection)
    records = []
    for item in chosen:
        row = table[item["row"]]
        record: dict[str, Any] = {"id": item["id"]}
        if "prompt" in roles:
            record["prompt"] = row[roles["prompt"]]
        if "image" in roles:
            record["image"] = _encoded_media(row[roles["image"]], ".png")
        if "audio" in roles:
            audio, rate = soundfile.read(io.BytesIO(row[roles["audio"]]["bytes"]), dtype="float32")
            buffer = io.BytesIO()
            soundfile.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
            record["audio"] = {"$file": {"suffix": ".wav", "b64": base64.b64encode(buffer.getvalue()).decode()}}
        if label_column:
            record["label"] = row[label_column]
        if source.get("label_image"):
            record["label"] = {"png_b64": base64.b64encode(row[label_column]["bytes"]).decode()}
        if source.get("polygon_label") and "image" in roles:  # an object mask as a polygon, with a point inside it
            from PIL import Image

            with Image.open(io.BytesIO(row[roles["image"]]["bytes"])) as image:
                size = list(image.size)
            polygon = row[source["polygon_label"]]
            record["label"] = {"polygon": polygon, "image_size": size}
            record["point_x"], record["point_y"] = _interior_point(polygon, size)
        if source.get("label_with_image_size") and "image" in roles:  # e.g. a box scored in normalized units
            from PIL import Image

            with Image.open(io.BytesIO(row[roles["image"]]["bytes"])) as image:
                record["label"] = {"value": record.get("label"), "image_size": list(image.size)}
        for column in source.get("columns", []):
            record[column] = row[column]
        if source.get("task_column"):  # per-sample category (a metric may grade categories differently)
            record["task"] = row[source["task_column"]]
        records.append(record)
    return records


def _encoded_media(value: Mapping[str, Any], default_suffix: str) -> dict[str, Any]:
    from PIL import Image

    data = value["bytes"]
    with Image.open(io.BytesIO(data)) as image:
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format or "", default_suffix)
    return {"$file": {"suffix": suffix, "b64": base64.b64encode(data).decode()}}


def _field(record: Mapping[str, Any], name: str) -> Any:
    """A record field; dotted names reach into nested objects and lists (``media.0.path``)."""
    value: Any = record
    for part in name.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _verified(source: Mapping[str, Any], environment: Environment) -> Path:
    path = environment.path("data_root") / require(source, "path", "source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != require(source, "sha256", "source"):
        raise ConfigError(f"{path} sha256 {digest} does not match the suite definition")
    return path


def _json_manifest_records(source: Mapping[str, Any], environment: Environment) -> list[dict[str, Any]]:
    """Records from a sha256-verified JSON, JSONL, or TSV manifest under data_root.

    ``file_fields`` (dotted names reach nested values) are resolved relative to the manifest
    directory, so ``*_path`` request fields are inlined (``square_crop`` fields become their centered
    square once selected); ``explode`` turns each record into one record per listed text field.
    """
    path = _verified(source, environment)
    if path.suffix == ".jsonl":
        data: Any = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".tsv":
        data = list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))
    else:
        data = json.loads(path.read_text())
    records = data[source["records"]] if source.get("records") else data
    id_field = source.get("id_field", "id")
    records = [{**record, "id": str(record[id_field]) if id_field in record else str(index)}
               for index, record in enumerate(records)]
    for field in source.get("file_fields", []):
        for record in records:
            record[field] = str(path.parent / _field(record, field))
    explode = source.get("explode")
    if explode:
        records = [{**record, "id": f"{record['id']}:{field}", "text": record[field]}
                   for record in records for field in explode]
    extract = source.get("extract")
    if extract:  # a field rewritten to the first group of a pattern (e.g. the sentence inside a chat prompt)
        import re

        pattern = re.compile(extract["pattern"], re.DOTALL)
        records = [{**record, extract["field"]: pattern.search(record[extract["field"]]).group(1)} for record in records]
    groups = source.get("retrieval_groups")
    if groups:  # each query, then its candidate documents, as consecutive text samples
        records = [item for record in records for item in (
            [{"id": f"{record['id']}:query", "text": record[groups["query"]], "task": "query",
              "label": list(record[groups["relevant"]])}]
            + [{"id": f"{record['id']}:doc{index}", "text": text, "task": "document"}
               for index, text in enumerate(record[groups["documents"]])])]
    if source.get("label_field"):  # a gold label for absolute-accuracy scoring
        records = [{**record, "label": _field(record, source["label_field"])} for record in records]
    if source.get("label_fields"):  # a gold label made of several fields (e.g. a program's tests)
        records = [{**record, "label": {name: record[name] for name in source["label_fields"]}} for record in records]
    return records


def _square_crop(path: Path, cache: Path) -> Path:
    """The centered square of an image (cached by content), so models that resize their condition
    image to a fixed square and models that keep its aspect ratio see the same input."""
    from PIL import Image

    target = cache / "trtmc-derived" / f"{hashlib.sha256(path.read_bytes()).hexdigest()[:24]}-square.png"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            side = min(rgb.size)
            left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
            rgb.crop((left, top, left + side, top + side)).save(target)
    return target


def _etth1_window_records(source: Mapping[str, Any], environment: Environment) -> list[dict[str, Any]]:
    """Seeded ETTh1 forecast windows, as benchmark qualification draws them: every ``stride`` hours of
    the test range, shuffled with ``seed``; ``past_values`` is row-major [time, column]."""
    rows = list(csv.DictReader(io.StringIO(_verified(source, environment).read_text())))
    window = require(source, "window", "source")
    columns, context = window.get("columns", ["OT"]), int(window.get("context_length", 512))
    start, end = int(window.get("test_target_start", 11520)), int(window.get("test_end", 14400))
    prediction = int(window.get("prediction_length", 0))
    starts = list(range(start - context, end - context - prediction + 1, int(window.get("stride", 24))))
    random.Random(int(source.get("seed", 20260715))).shuffle(starts)
    if len(rows) < end or not starts:
        raise ConfigError("ETTh1 data cannot satisfy the configured window")
    records = []
    for index, first in enumerate(starts):
        values = [float(row[column]) for row in rows[first:first + context] for column in columns]
        record = {"id": f"etth1-{index:04d}", "request": {
            "past_values": values, "observed_mask": [1.0] * len(values), "frequency": int(window.get("frequency", 0))}}
        if source.get("gold"):  # the observed future the forecast is scored against (row-major [time, column])
            future = rows[first + context:first + context + prediction]
            record["label"] = [float(row[column]) for row in future for column in columns]
        records.append(record)
    return records


# Generation controls that resolve to -1 ("model default") unless stated under the name the testcase
# resolver reads: the descriptor of a family request keeps ``num_steps``, while image-generation
# testcases read ``num_inference_steps``. TRTMC then applies the family's default and a Diffusers
# reference the pipeline's (FLUX.1-schnell: 4 vs 28 steps), so both would time different work.
MODEL_DEFAULT_FIELDS = {"num_steps": ("num_steps", "num_inference_steps", "num_sampling_steps"),
                        "guidance_scale": ("guidance_scale",), "cfg_scale": ("cfg_scale",),
                        "num_frames": ("num_frames", "video_num_frames")}


def fill_model_defaults(request: Mapping[str, Any], *sources: Mapping[str, Any]) -> dict[str, Any]:
    """``request`` with each -1 generation control taken from the first source that states it."""
    unset = (-1, -1.0)
    filled = dict(request)
    for field, names in MODEL_DEFAULT_FIELDS.items():
        if request.get(field) not in unset:
            continue
        stated = [source[name] for source in sources for name in names if source.get(name) not in (None, *unset)]
        if stated:
            filled[field] = stated[0]
    return filled


def _qualification_perf_records(source: Mapping[str, Any], environment: Environment) -> list[dict[str, Any]]:
    """The request the family's performance qualification times, resolved like a catalog testcase."""
    from .models import _import_repository

    repo = environment.path("repo")
    _import_repository(repo)
    from qualification_tests.benchmark_qualification import accuracy, catalog, runtime

    profile = require(source, "profile", "source")
    case = next((case for case in catalog.discover(repo) if case.model == profile and case.kind == "performance"),
                None)
    if case is None or not isinstance(case.values.get("request"), Mapping):
        raise ConfigError(f"{profile} has no family performance request")
    request = accuracy._resolve_task_assets(case, case.values["request"])
    with tempfile.TemporaryDirectory() as scratch:
        descriptor = runtime.write_model_descriptor(case, Path(scratch), request)
        records = _catalog_testcase_records({"profile": str(descriptor)}, environment)
    resolved = records[0]["request"]
    if any(resolved.get(field) in (-1, -1.0) for field in MODEL_DEFAULT_FIELDS):
        catalog_request = _catalog_testcase_records({"profile": profile}, environment)[0]["request"]
        resolved = fill_model_defaults(resolved, request, catalog_request)
    return [{**records[0], "request": resolved, "id": f"{profile}:{case.name}"}]


def _catalog_testcase_records(source: Mapping[str, Any], environment: Environment) -> list[dict[str, Any]]:
    """The catalog testcase request (``profile`` is a catalog name or a model descriptor file),
    resolved by trtmc-perf-serve exactly as trtmc-bench does."""
    repo = environment.path("repo")
    env = {**os.environ, "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": f"{repo}/apps/perf_serving:{repo}/apps/benchmark:{repo}/core/builder:{repo}"}
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "payload.jsonl"
        command = [str(environment["serve_python"]), "-m", "trtmc_perf_serving", "payload", "--manifest-root",
                   str(repo / "families"), "--profile", require(source, "profile", "source"), "--output", str(output)]
        if source.get("testcase"):
            command += ["--testcase", source["testcase"]]
        subprocess.run(command, check=True, cwd=repo, env=env, capture_output=True, text=True)
        request = json.loads(output.read_text())["payload"]["request"]
    return [{"id": source.get("testcase") or source["profile"], "request": request}]


_TOKENIZERS: dict[tuple[str, str | None], Any] = {}


def _truncate_prompt(prompt: str, truncation: Mapping[str, Any]) -> str:
    """Keep the last ``max_tokens`` prompt tokens of the model's tokenizer (left truncation).

    Both backends receive the same truncated text, so parity is unaffected; the bound keeps the
    prompt plus generated tokens inside the bundle's sequence limit.
    """
    key = (truncation["tokenizer"], truncation.get("revision"))
    if key not in _TOKENIZERS:
        from transformers import AutoTokenizer

        try:
            _TOKENIZERS[key] = AutoTokenizer.from_pretrained(
                key[0], revision=key[1], trust_remote_code=bool(truncation.get("trust_remote_code")))
        except Exception:  # noqa: BLE001 - fall back to a conservative character bound
            _TOKENIZERS[key] = None
    tokenizer, limit = _TOKENIZERS[key], int(truncation["max_tokens"])
    if tokenizer is None:
        return prompt[-limit * 2:]
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return prompt if len(ids) <= limit else tokenizer.decode(ids[-limit:])


def build_suite(definition: Mapping[str, Any], environment: Environment) -> Suite:
    from trtmc_perf_serving.files import inline_files

    source = definition["source"]
    kind = require(source, "kind", "source")
    selection = definition["selection"]
    if kind == "mmlu":
        records = _mmlu_records(source)
    elif kind == "aiperf_public":
        records = _aiperf_public_records(source, selection)
        selection = {"method": "first", "count": len(records)}  # already selected before decoding
    elif kind == "hf_dataset":
        records = _hf_dataset_records(source, selection)
        selection = {"method": "first", "count": len(records)}  # already selected before decoding
    elif kind == "image_archive":
        records = _image_archive_records(source, environment)
    elif kind == "url_jsonl":
        records = _url_jsonl_records(source, environment)
    elif kind == "json_manifest":
        records = _json_manifest_records(source, environment)
    elif kind == "catalog_testcase":
        records = _catalog_testcase_records(source, environment)
    elif kind == "qualification_perf":
        records = _qualification_perf_records(source, environment)
    elif kind == "etth1_windows":
        records = _etth1_window_records(source, environment)
    elif kind == "inline":
        records = [dict(record) for record in require(source, "records", "source")]
    else:
        raise ConfigError(f"unknown suite source kind {kind!r}")
    # base_profile: the profile's catalog testcase request is the base every sample overrides
    # (for example a generation's size, steps, and seed around a dataset prompt).
    base = (_catalog_testcase_records({"profile": definition["base_profile"]}, environment)[0]["request"]
            if definition.get("base_profile") else {})
    samples = []
    for record in select(records, selection):
        for field in source.get("square_crop", []):  # only the selected images
            record = {**record, field: str(_square_crop(Path(record[field]), Path(environment["hf_datasets_cache"])))}
        request = {**base, **(record.get("request") or {})}
        request.update(definition.get("request") or {})
        for source_field, request_field in (definition.get("fields") or {}).items():
            request[request_field] = record[source_field] if source_field in record else _field(record, source_field)
        if isinstance(request.get("prompt"), str) and (definition.get("prompt_prefix") or definition.get("prompt_suffix")):
            request["prompt"] = f"{definition.get('prompt_prefix', '')}{request['prompt']}{definition.get('prompt_suffix', '')}"
        if definition.get("truncate_prompt") and isinstance(request.get("prompt"), str):
            request["prompt"] = _truncate_prompt(request["prompt"], definition["truncate_prompt"])
        request = inline_files(request)
        sample = {"sample_id": record["id"], "task": record.get("task", definition["suite"]),
                  "request": request, "request_sha": request_sha(request)}
        if "label" in record:
            sample["label"] = record["label"]
        samples.append(sample)
    key = sha256_text(canonical({"suite": definition["suite"], "version": definition["version"],
                                 "samples": [sample["request_sha"] for sample in samples]}))
    manifest = {"suite": definition["suite"], "version": definition["version"], "key": key,
                "source": dict(source), "selection": dict(definition["selection"]), "samples": len(samples),
                **({"base_profile": definition["base_profile"]} if definition.get("base_profile") else {})}
    return Suite(definition["suite"], key, samples, manifest)
