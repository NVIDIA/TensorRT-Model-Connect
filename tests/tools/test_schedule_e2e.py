"""Tests for the E2E parallel scheduler."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import textwrap
from pathlib import Path

from tests.e2e_harness.manifest_loader import find_manifest_path

_SCHEDULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "schedule_e2e.py"
_SPEC = importlib.util.spec_from_file_location("schedule_e2e", _SCHEDULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
schedule_e2e = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(schedule_e2e)


def _write_manifest(manifest_dir: Path, name: str, **fields: object) -> None:
    manifest = {
        "name": name,
        "hf_id": f"org/{name}",
        "runtime_strategy": f"{name.replace('-', '_')}_decoder_kv_cache",
        **fields,
    }
    (manifest_dir / f"{name}.json").write_text(json.dumps(manifest))


def _test_id(name: str) -> str:
    return f"tests/e2e/models/unit_family/test_unit_family_e2e.py::test_model_e2e[{name}]"


def test_diffusion_family_strategies_are_large() -> None:
    assert schedule_e2e.classify_size({"runtime_strategy": "diffusion"}) == "large"
    assert schedule_e2e.classify_size({"runtime_strategy": "diffusion_primary"}) == "large"
    assert schedule_e2e.classify_size({"runtime_strategy": "diffusion_secondary"}) == "large"


def test_manifest_size_override_is_authoritative() -> None:
    assert schedule_e2e.classify_size({"e2e_size": "large"}) == "large"
    assert schedule_e2e.classify_size({
        "runtime_strategy": "diffusion_primary",
        "e2e_size": "small",
    }) == "small"


def test_exclusive_gpu_resource_reserves_gpu(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "exclusive-media-a",
        runtime_strategy="diffusion_primary",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(
        tmp_path,
        "exclusive-media-b",
        runtime_strategy="diffusion_primary",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(
        tmp_path,
        "exclusive-media-c",
        runtime_strategy="diffusion_primary",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(tmp_path, "small-a")
    _write_manifest(tmp_path, "small-b")
    _write_manifest(tmp_path, "large-a", hf_id="org/model-9B")

    assignments = schedule_e2e.schedule(
        [
            _test_id("exclusive-media-a"),
            _test_id("exclusive-media-b"),
            _test_id("exclusive-media-c"),
            _test_id("small-a"),
            _test_id("small-b"),
            _test_id("large-a"),
        ],
        tmp_path,
        num_gpus=4,
        workers_per_gpu=2,
    )

    assert assignments["0"] == [[_test_id("exclusive-media-a")]]
    assert assignments["1"] == [[_test_id("exclusive-media-b")]]
    assert assignments["2"] == [[_test_id("exclusive-media-c")]]
    assert assignments["3"]
    shared_tests = [test for worker in assignments["3"] for test in worker]
    assert sorted(shared_tests) == sorted([
        _test_id("small-a"),
        _test_id("small-b"),
        _test_id("large-a"),
    ])


def test_same_bundle_exclusive_tests_stay_in_one_worker_queue(tmp_path: Path) -> None:
    for mode in ("mode-a", "mode-b", "mode-c", "mode-d"):
        _write_manifest(
            tmp_path,
            f"shared-exclusive-{mode}",
            runtime_strategy="diffusion_text_experiment",
            e2e_parallel_resource="exclusive_gpu",
            bundle="shared-exclusive-bundle.trtfb",
            metadata={"group_by_bundle": True},
        )
    _write_manifest(
        tmp_path,
        "other-exclusive",
        runtime_strategy="diffusion_primary",
        e2e_parallel_resource="exclusive_gpu",
    )

    grouped_ids = [
        _test_id("shared-exclusive-mode-a"),
        _test_id("shared-exclusive-mode-b"),
        _test_id("shared-exclusive-mode-c"),
        _test_id("shared-exclusive-mode-d"),
    ]
    assignments = schedule_e2e.schedule(
        [*grouped_ids, _test_id("other-exclusive")],
        tmp_path,
        num_gpus=4,
        workers_per_gpu=1,
    )

    queues = [worker for workers in assignments.values() for worker in workers]
    matching_queues = [
        worker for worker in queues
        if any(test_id in worker for test_id in grouped_ids)
    ]
    assert len(matching_queues) == 1
    assert sorted(matching_queues[0]) == sorted(grouped_ids)


def test_same_bundle_shared_tests_stay_in_one_worker_queue(tmp_path: Path) -> None:
    for mode in ("a", "b", "c"):
        _write_manifest(
            tmp_path,
            f"shared-mode-{mode}",
            bundle="shared-bundle.trtfb",
            metadata={"group_by_bundle": True},
        )
    _write_manifest(tmp_path, "small-a")
    _write_manifest(tmp_path, "small-b")

    grouped_ids = [
        _test_id("shared-mode-a"),
        _test_id("shared-mode-b"),
        _test_id("shared-mode-c"),
    ]
    assignments = schedule_e2e.schedule(
        [*grouped_ids, _test_id("small-a"), _test_id("small-b")],
        tmp_path,
        num_gpus=1,
        workers_per_gpu=3,
    )

    queues = assignments["0"]
    matching_queues = [
        worker for worker in queues
        if any(test_id in worker for test_id in grouped_ids)
    ]
    assert len(matching_queues) == 1
    assert sorted(matching_queues[0]) == sorted(grouped_ids)


def test_same_bundle_tests_without_group_metadata_stay_separate(tmp_path: Path) -> None:
    for mode in ("a", "b", "c"):
        _write_manifest(
            tmp_path,
            f"plain-shared-mode-{mode}",
            bundle="shared-bundle.trtfb",
        )

    grouped_ids = [
        _test_id("plain-shared-mode-a"),
        _test_id("plain-shared-mode-b"),
        _test_id("plain-shared-mode-c"),
    ]
    groups = schedule_e2e._group_by_bundle(
        grouped_ids,
        schedule_e2e._load_manifests(tmp_path),
    )

    assert [group for _, group in groups] == [[test_id] for test_id in grouped_ids]


def test_bundle_group_entry_summarizes_member_testcases(tmp_path: Path) -> None:
    for mode in ("a", "b", "c"):
        _write_manifest(
            tmp_path,
            f"shared-mode-{mode}",
            bundle="shared-bundle.trtfb",
        )
    _write_manifest(tmp_path, "unique-mode", bundle="unique-bundle.trtfb")

    summary = schedule_e2e.bundle_selection_summary(
        [
            _test_id("bundle:shared-mode-a+shared-mode-b+shared-mode-c"),
            _test_id("unique-mode"),
        ],
        tmp_path,
    )

    assert summary == {
        "selected_entries": 2,
        "selected_testcases": 4,
        "unique_bundle_identities": 2,
        "shared_bundle_groups": 1,
        "testcases_in_shared_bundle_groups": 3,
        "missing_manifests": 0,
    }


def test_bundle_group_entry_weight_sums_members(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "small-a", bundle="shared.trtfb")
    _write_manifest(tmp_path, "large-b", hf_id="org/model-9B", bundle="shared.trtfb")
    manifests = schedule_e2e._load_manifests(tmp_path)

    assert schedule_e2e._test_weight(
        _test_id("bundle:small-a+large-b"),
        manifests,
        timing_estimates={"small-a": 10.0, "large-b": 20.0},
    ) == 30.0
    assert schedule_e2e._test_weight(
        _test_id("bundle:small-a+large-b"),
        manifests,
        timing_estimates=None,
    ) == (
        schedule_e2e._SMALL_TEST_WEIGHT
        + schedule_e2e._LARGE_TEST_WEIGHT
    )


def test_phase_schedule_keeps_shared_workers_after_exclusive_gpus(tmp_path: Path) -> None:
    for name in ("exclusive-a", "exclusive-b", "exclusive-c"):
        _write_manifest(
            tmp_path,
            name,
            runtime_strategy="diffusion_primary",
            e2e_parallel_resource="exclusive_gpu",
        )
    for name in ("small-a", "small-b", "small-c", "small-d", "small-e", "small-f"):
        _write_manifest(tmp_path, name)

    phases = schedule_e2e.schedule_phases(
        [
            _test_id("exclusive-a"),
            _test_id("exclusive-b"),
            _test_id("exclusive-c"),
            _test_id("small-a"),
            _test_id("small-b"),
            _test_id("small-c"),
            _test_id("small-d"),
            _test_id("small-e"),
            _test_id("small-f"),
        ],
        tmp_path,
        num_gpus=2,
        workers_per_gpu=2,
    )

    assert [phase["name"] for phase in phases] == ["exclusive_gpu", "shared"]
    exclusive_schedule = phases[0]["schedule"]
    shared_schedule = phases[1]["schedule"]

    exclusive_tests = [
        test for workers in exclusive_schedule.values() for worker in workers for test in worker
    ]
    shared_tests = [
        test for workers in shared_schedule.values() for worker in workers for test in worker
    ]

    assert sorted(exclusive_tests) == sorted([
        _test_id("exclusive-a"),
        _test_id("exclusive-b"),
        _test_id("exclusive-c"),
    ])
    assert sorted(shared_tests) == sorted([
        _test_id("small-a"),
        _test_id("small-b"),
        _test_id("small-c"),
        _test_id("small-d"),
        _test_id("small-e"),
        _test_id("small-f"),
    ])
    assert sum(len(workers) for workers in exclusive_schedule.values()) == 2
    assert sum(len(workers) for workers in shared_schedule.values()) == 2


def test_phase_schedule_offsets_large_shared_work_by_exclusive_load(tmp_path: Path) -> None:
    for idx in range(5):
        _write_manifest(
            tmp_path,
            f"exclusive-{idx}",
            runtime_strategy="diffusion_primary",
            e2e_parallel_resource="exclusive_gpu",
        )
    for idx in range(9):
        _write_manifest(tmp_path, f"large-{idx}", hf_id=f"org/large-{idx}-4B")

    phases = schedule_e2e.schedule_phases(
        [
            *[_test_id(f"exclusive-{idx}") for idx in range(5)],
            *[_test_id(f"large-{idx}") for idx in range(9)],
        ],
        tmp_path,
        num_gpus=3,
        workers_per_gpu=2,
    )

    shared_schedule = phases[1]["schedule"]
    shared_counts = {
        gpu_id: sum(len(worker) for worker in workers)
        for gpu_id, workers in shared_schedule.items()
    }

    assert shared_counts == {"0": 2, "1": 1, "2": 6}


def test_large_and_small_tests_are_balanced_across_even_worker_count(tmp_path: Path) -> None:
    large_names = [f"large-{idx}" for idx in range(8)]
    small_names = [f"small-{idx}" for idx in range(8)]
    for name in large_names:
        _write_manifest(tmp_path, name, hf_id=f"org/{name}-4B")
    for name in small_names:
        _write_manifest(tmp_path, name)

    assignments = schedule_e2e.schedule(
        [_test_id(name) for name in [*large_names, *small_names]],
        tmp_path,
        num_gpus=1,
        workers_per_gpu=4,
    )

    workers = assignments["0"]
    assert len(workers) == 4
    for worker in workers:
        worker_names = {_test_id(name) for name in large_names}
        large_count = sum(1 for test_id in worker if test_id in worker_names)
        assert large_count == 2
        assert len(worker) == 4


def test_run_e2e_parallel_pipelines_exclusive_then_shared_work(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'GPU 0: fake\\nGPU 1: fake\\n'\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)

    fake_python = bin_dir / "fake-python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import html
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            if args and args[0] == "-":
                sys.stdin.read()
                if len(args) == 2:
                    raise SystemExit("models-file collection is not used by this test")
                raise SystemExit(0)

            if len(args) >= 2 and args[:2] == ["-m", "pytest"]:
                if "--co" in args:
                    models_file = None
                    if "--e2e-models-file" in args:
                        idx = args.index("--e2e-models-file")
                        models_file = args[idx + 1]
                    if models_file is None:
                        raise SystemExit("--e2e-models-file missing")
                    for line in Path(models_file).read_text(encoding="utf-8").splitlines():
                        model = line.strip()
                        if model:
                            print(
                                "tests/e2e/models/fake_family/"
                                f"test_fake_family_e2e.py::test_model_e2e[{model}]"
                            )
                    raise SystemExit(0)

                tests = [
                    arg for arg in args
                    if arg.startswith("tests/")
                    and (
                        "::test_e2e[" in arg
                        or "::test_model_e2e[" in arg
                    )
                ]
                junit_path = None
                for arg in args:
                    if arg.startswith("--junitxml="):
                        junit_path = arg.split("=", 1)[1]
                        break
                for test in tests:
                    print(f"{test} PASSED")
                print(f"=== {len(tests)} passed in 0.01s ===")
                if junit_path:
                    cases = "".join(
                        f'<testcase classname="e2e" name="{html.escape(test)}" />'
                        for test in tests
                    )
                    Path(junit_path).write_text(
                        f'<testsuites><testsuite tests="{len(tests)}">'
                        f"{cases}</testsuite></testsuites>",
                        encoding="utf-8",
                    )
                raise SystemExit(0)

            raise SystemExit(f"unexpected fake-python args: {args}")
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    _write_manifest(
        manifest_dir,
        "exclusive-media-a",
        runtime_strategy="diffusion_primary",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(
        manifest_dir,
        "exclusive-media-b",
        runtime_strategy="diffusion_primary",
        e2e_parallel_resource="exclusive_gpu",
    )
    for name, family in {
        "encoder-a": "encoder_family",
        "encoder-b": "encoder_family",
        "decoder-a": "decoder_family",
        "decoder-b": "decoder_family",
    }.items():
        _write_manifest(manifest_dir, name, family=family)

    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(
        "\n".join([
            "tests/e2e/models/media_family/test_media_family_e2e.py::test_model_e2e[exclusive-media-a]",
            "tests/e2e/models/media_family/test_media_family_e2e.py::test_model_e2e[exclusive-media-b]",
            "tests/e2e/models/encoder_family/test_encoder_family_e2e.py::test_model_e2e[encoder-a]",
            "tests/e2e/models/encoder_family/test_encoder_family_e2e.py::test_model_e2e[encoder-b]",
            "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-a]",
            "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-b]",
            "",
        ]),
        encoding="utf-8",
    )

    result_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["TRTMC_E2E_EXCLUDE_GPU0"] = "0"
    env["TRTMC_E2E_MANIFEST_DIR"] = str(manifest_dir)

    completed = subprocess.run(
        [
            str(repo_root / "scripts" / "run_e2e_parallel.sh"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--result-dir",
            str(result_dir),
            "--trtmc-binary",
            str(tmp_path / "trtmc"),
            "--hf-python",
            str(fake_python),
            "--num-gpus",
            "2",
            "--workers-per-gpu",
            "2",
            "--tests-file",
            str(tests_file),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    assert "=== E2E pipelined phases: exclusive_gpu -> shared ===" in completed.stdout
    assert "Workers planned:" in completed.stdout
    assert "gpu0-exclusive_gpu-w0" in completed.stdout
    assert "gpu1-shared-w0" in completed.stdout

    schedule = json.loads((result_dir / "schedule.json").read_text(encoding="utf-8"))
    assert [phase["name"] for phase in schedule["phases"]] == [
        "exclusive_gpu",
        "shared",
    ]
    exclusive_phase, shared_phase = schedule["phases"]
    exclusive_tests = {
        test
        for gpu_workers in exclusive_phase["schedule"].values()
        for worker_tests in gpu_workers
        for test in worker_tests
    }
    shared_tests = {
        test
        for gpu_workers in shared_phase["schedule"].values()
        for worker_tests in gpu_workers
        for test in worker_tests
    }
    assert exclusive_tests == {
        "tests/e2e/models/media_family/test_media_family_e2e.py::test_model_e2e[exclusive-media-a]",
        "tests/e2e/models/media_family/test_media_family_e2e.py::test_model_e2e[exclusive-media-b]",
    }
    assert shared_tests == {
        "tests/e2e/models/encoder_family/test_encoder_family_e2e.py::test_model_e2e[encoder-a]",
        "tests/e2e/models/encoder_family/test_encoder_family_e2e.py::test_model_e2e[encoder-b]",
        "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-a]",
        "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-b]",
    }
    assert len(list(result_dir.glob("console-gpu*-w*.log"))) == 6


def test_run_e2e_parallel_collects_grouped_entries_when_models_file_is_present(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    fake_python = bin_dir / "fake-python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import html
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            if args and args[0] == "-":
                sys.stdin.read()
                raise SystemExit(0)

            if len(args) >= 2 and args[:2] == ["-m", "pytest"]:
                if "--co" in args:
                    assert "--e2e-group-by-bundle" in args
                    assert "--e2e-models-file" in args
                    models_file = args[args.index("--e2e-models-file") + 1]
                    selected = set(
                        line.strip()
                        for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                    if {"group-a", "group-b"} <= selected:
                        print(
                            "tests/e2e/models/fake_family/"
                            "test_fake_family_e2e.py::test_model_e2e[bundle:group-a+group-b]"
                        )
                    if "solo" in selected:
                        print(
                            "tests/e2e/models/fake_family/"
                            "test_fake_family_e2e.py::test_model_e2e[solo]"
                        )
                    raise SystemExit(0)

                tests = [
                    arg for arg in args
                    if arg.startswith("tests/")
                    and (
                        "::test_e2e[" in arg
                        or "::test_model_e2e[" in arg
                    )
                ]
                stale = [test for test in tests if "[group-a]" in test or "[group-b]" in test]
                if stale:
                    raise SystemExit(f"stale ungrouped IDs scheduled: {stale}")

                junit_path = None
                for arg in args:
                    if arg.startswith("--junitxml="):
                        junit_path = arg.split("=", 1)[1]
                        break
                for test in tests:
                    print(f"{test} PASSED")
                print(f"=== {len(tests)} passed in 0.01s ===")
                if junit_path:
                    cases = "".join(
                        f'<testcase classname="e2e" name="{html.escape(test)}" />'
                        for test in tests
                    )
                    Path(junit_path).write_text(
                        f'<testsuites><testsuite tests="{len(tests)}">'
                        f"{cases}</testsuite></testsuites>",
                        encoding="utf-8",
                    )
                raise SystemExit(0)

            raise SystemExit(f"unexpected fake-python args: {args}")
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    _write_manifest(
        manifest_dir,
        "group-a",
        family="fake_family",
        bundle="shared.trtfb",
    )
    _write_manifest(
        manifest_dir,
        "group-b",
        family="fake_family",
        bundle="shared.trtfb",
    )
    _write_manifest(manifest_dir, "solo", family="fake_family")

    models_file = tmp_path / "models.txt"
    models_file.write_text("group-a\ngroup-b\nsolo\n", encoding="utf-8")
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(
        "\n".join([
            "tests/e2e/models/fake_family/test_fake_family_e2e.py::test_model_e2e[group-a]",
            "tests/e2e/models/fake_family/test_fake_family_e2e.py::test_model_e2e[group-b]",
            "tests/e2e/models/fake_family/test_fake_family_e2e.py::test_model_e2e[solo]",
            "",
        ]),
        encoding="utf-8",
    )

    result_dir = tmp_path / "results"
    env = os.environ.copy()
    env["NUM_GPUS"] = "1"
    env["TRTMC_E2E_EXCLUDE_GPU0"] = "0"
    env["TRTMC_E2E_MANIFEST_DIR"] = str(manifest_dir)

    completed = subprocess.run(
        [
            str(repo_root / "scripts" / "run_e2e_parallel.sh"),
            "--engine-dir",
            str(tmp_path / "engines"),
            "--result-dir",
            str(result_dir),
            "--trtmc-binary",
            str(tmp_path / "trtmc"),
            "--hf-python",
            str(fake_python),
            "--workers-per-gpu",
            "1",
            "--models-file",
            str(models_file),
            "--tests-file",
            str(tests_file),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    assert "Scheduler source: models file (grouped collection)" in completed.stdout

    schedule = json.loads((result_dir / "schedule.json").read_text(encoding="utf-8"))
    scheduled = {
        test
        for phase in schedule["phases"]
        for gpu_workers in phase["schedule"].values()
        for worker_tests in gpu_workers
        for test in worker_tests
    }
    assert scheduled == {
        "tests/e2e/models/fake_family/test_fake_family_e2e.py::test_model_e2e[bundle:group-a+group-b]",
        "tests/e2e/models/fake_family/test_fake_family_e2e.py::test_model_e2e[solo]",
    }


def test_gpt_oss_20b_is_marked_exclusive_gpu() -> None:
    manifest_dir = Path(__file__).resolve().parents[1] / "e2e" / "models"

    for manifest_name in ("gpt-oss-20b", "gpt-oss-20b-l0"):
        manifest_path = find_manifest_path(manifest_name, manifest_dir)
        assert manifest_path is not None
        manifest = json.loads(manifest_path.read_text())
        assert schedule_e2e.classify_parallel_resource(manifest) == "exclusive_gpu"
