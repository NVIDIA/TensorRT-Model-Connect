# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.validation import catalog as validation_catalog


def test_manifest_records_can_be_loaded_by_canonical_name(monkeypatch) -> None:
    models_dir = Path("models")
    records = [
        {"name": "model-b", "family": "family-b"},
        {"name": "model-a", "family": "family-a"},
    ]
    monkeypatch.setattr(
        validation_catalog,
        "load_manifest_records",
        lambda path: records if path == models_dir else [],
    )

    assert validation_catalog.load_manifest_records_by_name(models_dir) == {
        "model-b": records[0],
        "model-a": records[1],
    }


def test_model_owned_validation_dataset_is_merged_for_selected_workload(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "demo"
    manifests_dir = model_dir / "manifests"
    manifests_dir.mkdir(parents=True)
    (model_dir / "MODEL.toml").write_text(
        'id = "demo"\n'
        'test_manifests = ["manifests/demo.json"]\n'
        "\n"
        "[validation_datasets.demo_parity]\n"
        'default_path = "/mnt/data/demo/dataset.json"\n'
        'input_asset_fields = ["image"]\n',
        encoding="utf-8",
    )
    manifest_path = manifests_dir / "demo.json"
    manifest_path.write_text(
        '{"name":"demo-model","family":"demo",'
        '"runtime_strategy":"demo_runtime","task_strategy":"demo_task"}\n',
        encoding="utf-8",
    )

    model = validation_catalog.manifest_record(manifest_path)
    suite = {
        "id": "demo_parity",
        "dataset": {"kind": "model_plugin_json"},
    }

    assert validation_catalog.resolve_suite_for_model(suite, model)["dataset"] == {
        "kind": "model_plugin_json",
        "default_path": "/mnt/data/demo/dataset.json",
        "input_asset_fields": ["image"],
    }
    assert suite["dataset"] == {"kind": "model_plugin_json"}
