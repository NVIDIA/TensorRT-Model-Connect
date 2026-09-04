# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import io
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.ci.context import CiContext
from tools.ci.container import CiContainer
from tools.ci.docker_image import DockerImageManager
from tools.ci.e2e import E2ERunner, _require_passing_junit
from tools.ci.package import (
    SourceArchiveValidator,
    WheelArchiveValidator,
    WheelPackageManager,
    load_native_libraries,
)
from tools.ci.pipeline import CiPipeline
from tools.ci.process import CiError
from tools.ci.quality import SourceQualityChecks, UnitTestRunner
from tools import perf_matrix


class RecordingContext:
    def __init__(self, repository: Path, env: dict[str, str]):
        self.repository = repository
        self.env = env
        self.calls = []
        self.runtime_snapshots = []
        self.skip_ctest = False
        self.no_cpu_family_tests = False

    def run(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        runtime_root = kwargs.get("updates", {}).get("TRTMC_RUNTIME_ROOT")
        if runtime_root:
            self.runtime_snapshots.append(
                tuple(sorted(path.name for path in Path(runtime_root).iterdir()))
            )
        if "--show-only=json-v1" in command:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "tests": [
                            {
                                "name": "beta_runtime",
                                "properties": [
                                    {
                                        "name": "WORKING_DIRECTORY",
                                        "value": str(
                                            Path(self.env["TRTMC_NATIVE_BUILD_DIR"])
                                            / "families/beta"
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                )
            )
        if "--output-junit" in command:
            report = Path(command[command.index("--output-junit") + 1])
            outcome = '<skipped message="not run" />' if self.skip_ctest else ""
            report.write_text(
                f'<testsuites><testsuite><testcase name="beta_runtime">{outcome}'
                "</testcase></testsuite></testsuites>",
                encoding="utf-8",
            )
        if "--junitxml" in command:
            report = Path(command[command.index("--junitxml") + 1])
            empty = self.no_cpu_family_tests and "-unit-junit.xml" in report.name
            report.write_text(
                "<testsuites><testsuite>"
                + ("" if empty else '<testcase name="hardware" />')
                + "</testsuite></testsuites>",
                encoding="utf-8",
            )
            if empty:
                return SimpleNamespace(stdout="", returncode=5)
        return SimpleNamespace(stdout="", returncode=0)


@pytest.mark.parametrize(
    ("testcases", "selector"),
    (
        (None, ["--e2e-model", "beta"]),
        (["beta-case"], ["--e2e-testcase", "beta-case"]),
    ),
)
def test_selective_e2e_calls_family_tests_directly(
    tmp_path: Path,
    testcases: list[str] | None,
    selector: list[str],
) -> None:
    for family in ("alpha", "beta"):
        root = tmp_path / "families" / family
        (root / "tests").mkdir(parents=True)
        (root / "model.py").write_text("def build(request, writer): pass\n")
        (root / "tests/test_e2e.py").write_text("def test_e2e(): pass\n")
        if family == "beta":
            (root / "tests/test_model.py").write_text(
                "def test_model(): pass\n",
                encoding="utf-8",
            )
            (root / "tests/test_gpu.py").write_text(
                "import pytest\n\n@pytest.mark.gpu\ndef test_gpu(): pass\n",
                encoding="utf-8",
            )
    binary = tmp_path / "trtmc"
    binary.write_text("")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "libtrtmc_core.so").write_text("")
    (runtime / "libtrtmc_backend_trt.so").write_text("")
    (runtime / "libtrtmc_model_beta.so").write_text("")
    native_build = tmp_path / "native-build"
    native_build.mkdir()
    (native_build / "CTestTestfile.cmake").write_text("")
    impact = {"families": ["beta"]}
    if testcases is not None:
        impact["testcases"] = testcases
    (tmp_path / "impact.json").write_text(json.dumps(impact))
    context = RecordingContext(
        tmp_path,
        {
            "TRTMC_BINARY": str(binary),
            "TRTMC_RUNTIME_ROOT": str(runtime),
            "TRTMC_NATIVE_BUILD_DIR": str(native_build),
        },
    )

    E2ERunner(context).selective()

    assert context.calls[0][0][:3] == ["ctest", "--test-dir", native_build]
    assert context.calls[1][0][-1] == "test_beta_runtime"
    assert context.calls[2][0][0] == "ctest"
    assert "--output-junit" in context.calls[2][0]
    unit_command, unit_options = context.calls[3]
    assert unit_command[:5] == [
        "python",
        "-m",
        "pytest",
        "families/beta/tests/test_gpu.py",
        "families/beta/tests/test_model.py",
    ]
    assert ["-m", "not gpu and not trt"] == unit_command[5:7]
    assert unit_command[-2:] == [
        "--junitxml",
        native_build / "trtmc-beta-unit-junit.xml",
    ]
    assert unit_options["unset"] == ("PYTEST_ADDOPTS",)
    assert unit_options["check"] is False

    hardware_command, _ = context.calls[4]
    assert hardware_command[:4] == [
        "python",
        "-m",
        "pytest",
        "families/beta/tests/test_gpu.py",
    ]
    assert ["-m", "gpu or trt"] == hardware_command[4:6]
    assert hardware_command[-2:] == [
        "--junitxml",
        native_build / "trtmc-beta-hardware-junit.xml",
    ]
    assert context.calls[4][1]["unset"] == ("PYTEST_ADDOPTS",)

    command, options = context.calls[5]
    assert command[:4] == ["python", "-m", "pytest", "families/beta/tests/test_e2e.py"]
    assert selector == command[4:6]
    rendered = " ".join(command)
    assert "e2e_harness" not in rendered
    assert "--trtmc-binary" not in rendered
    assert "--model-plugin-dir" not in rendered
    assert options["updates"]["TRTMC_BINARY"] == str(binary)
    assert options["updates"]["TRTMC_RUNTIME_ROOT"] != str(runtime)
    assert context.runtime_snapshots == [
        ("libtrtmc_backend_trt.so", "libtrtmc_core.so", "libtrtmc_model_beta.so")
    ]


def test_family_with_only_hardware_tests_accepts_exact_empty_cpu_result(
    tmp_path: Path,
) -> None:
    family = tmp_path / "families/beta"
    (family / "tests").mkdir(parents=True)
    (family / "model.py").write_text("def build(request, writer): pass\n")
    (family / "tests/test_e2e.py").write_text("def test_e2e(): pass\n")
    (family / "tests/test_gpu.py").write_text(
        "import pytest\n\n@pytest.mark.gpu\ndef test_gpu(): pass\n",
        encoding="utf-8",
    )
    binary = tmp_path / "trtmc"
    binary.write_text("")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in (
        "libtrtmc_core.so",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_beta.so",
    ):
        (runtime / name).write_text("")
    native_build = tmp_path / "native-build"
    native_build.mkdir()
    (native_build / "CTestTestfile.cmake").write_text("")
    context = RecordingContext(
        tmp_path,
        {
            "TRTMC_BINARY": str(binary),
            "TRTMC_RUNTIME_ROOT": str(runtime),
            "TRTMC_NATIVE_BUILD_DIR": str(native_build),
        },
    )
    context.no_cpu_family_tests = True

    E2ERunner(context)._run(("beta",))

    unit_command, unit_options = context.calls[3]
    assert ["-m", "not gpu and not trt"] == unit_command[4:6]
    assert unit_options["check"] is False
    assert context.calls[-1][0][:4] == [
        "python",
        "-m",
        "pytest",
        "families/beta/tests/test_e2e.py",
    ]


def test_family_cpp_skip_fails_closed(tmp_path: Path) -> None:
    family = tmp_path / "families/beta"
    family.mkdir(parents=True)
    build = tmp_path / "native-build"
    build.mkdir()
    context = RecordingContext(
        tmp_path,
        {"TRTMC_NATIVE_BUILD_DIR": str(build)},
    )
    context.skip_ctest = True

    with pytest.raises(CiError, match=r"family C\+\+ tests skipped"):
        E2ERunner(context)._run_family_ctests(build, ("beta",))


def test_family_hardware_skip_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "hardware.xml"
    report.write_text(
        '<testsuites><testsuite><testcase name="gpu"><skipped />'
        "</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )

    with pytest.raises(CiError, match="family hardware tests skipped: gpu"):
        _require_passing_junit(report, "family hardware tests")


@pytest.mark.parametrize(
    ("element", "outcome"),
    (("skipped", "skipped"), ("error", "errors"), ("failure", "failures")),
)
def test_family_python_unit_junit_fails_closed(
    tmp_path: Path,
    element: str,
    outcome: str,
) -> None:
    report = tmp_path / "unit.xml"
    report.write_text(
        f'<testsuites><testsuite><testcase name="unit"><{element} />'
        "</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )

    with pytest.raises(CiError, match=f"family Python unit tests {outcome}: unit"):
        _require_passing_junit(report, "family Python unit tests")


def test_e2e_rejects_multiple_family_environments(tmp_path: Path) -> None:
    context = RecordingContext(tmp_path, {})

    with pytest.raises(CiError, match="exactly one family"):
        E2ERunner(context)._run(("alpha", "beta"))


def test_pipeline_exposes_only_active_stages(tmp_path: Path) -> None:
    pipeline = CiPipeline(CiContext(tmp_path, {}))
    assert tuple(pipeline.stages) == (
        "impact",
        "family-coverage",
        "complexity",
        "lint",
        "source-quality",
        "premerge-unit",
        "package",
        "setup",
        "selective-e2e",
    )


def test_package_build_uses_the_preinstalled_offline_toolchain() -> None:
    source = inspect.getsource(WheelPackageManager.build)
    assert '"--no-isolation"' in source
    assert '"build>=1.2"' not in source
    repository = Path(__file__).resolve().parents[2]
    conanfile = (repository / "conanfile.py").read_text()
    assert "self.requires(" not in conanfile
    assert "CMakeDeps" not in conanfile
    dockerfile = (repository / "Dockerfile").read_text()
    assert "openmpi-bin" in dockerfile
    assert "nvidia/nccl/lib" in dockerfile


def test_source_quality_runs_complexity_before_other_checks() -> None:
    source = inspect.getsource(SourceQualityChecks.run)
    assert source.index("self.family_coverage()") < source.index("self.complexity()")
    assert source.index("self.complexity()") < source.index("self.lint_changed_files()")
    assert source.index("self.lint_changed_files()") < source.index("self.architecture_contracts()")


def test_source_quality_lints_only_files_that_still_exist(tmp_path: Path) -> None:
    (tmp_path / "kept.py").write_text("", encoding="utf-8")
    context = RecordingContext(tmp_path, {})
    context.run = lambda *_args, **_kwargs: SimpleNamespace(
        stdout="kept.py\ndeleted.py\n"
    )

    assert SourceQualityChecks(context)._changed_files("base", "*.py") == ["kept.py"]


def test_retired_central_ci_modules_are_absent() -> None:
    repository = Path(__file__).resolve().parents[2]
    retired = (
        "coverage.py",
        "e2e_schedule.py",
        "e2e_scheduler.py",
        "gpu_lease.py",
        "isolation.py",
        "model_proof.py",
        "model_proof_inner.py",
        "model_proof_selection.py",
        "model_reference_cache.py",
        "selected_wheel.py",
        "validation.py",
    )
    assert not [name for name in retired if (repository / "tools/ci" / name).exists()]
    assert not (repository / "scripts/schedule_e2e.py").exists()


def test_internal_bridge_waits_for_the_exact_run_until_the_job_timeout() -> None:
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/internal-ci-bridge.yml"
    source = workflow.read_text(encoding="utf-8")
    assert "timeout-minutes: 360" in source
    wait = source.split("- name: Wait for the exact Internal CI result", 1)[1].split(
        "\n  publish:", 1
    )[0]
    assert "while true; do" in wait
    assert "sleep 15" in wait
    assert "seq " not in wait
    assert "gh run download" not in source
    assert "tools.public_failure" not in source
    assert "policy_sha" not in source
    assert "Protected failure details are not transferred to the public repository." in source


def test_community_activity_alert_uses_only_trusted_external_metadata() -> None:
    path = (
        Path(__file__).resolve().parents[2] / ".github/workflows/community-activity-slack-alert.yml"
    )
    source = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    ready = workflow["jobs"]["notify-ready-pr"]
    activity = workflow["jobs"]["notify-activity"]

    assert workflow["permissions"] == {}
    assert ready["permissions"] == {
        "actions": "read",
        "checks": "read",
        "pull-requests": "read",
    }
    assert activity["permissions"] == {}
    assert ready["timeout-minutes"] == 7
    assert activity["timeout-minutes"] == 5
    assert all("uses" not in step for job in (ready, activity) for step in job["steps"])
    assert "actions/checkout" not in source

    condition = activity["if"]
    assert "github.repository == 'NVIDIA/TensorRT-Model-Connect'" in condition
    assert "github.event.sender.type != 'Bot'" in condition
    association_fields = {
        "issues": "github.event.issue.author_association",
        "issue_comment": "github.event.comment.author_association",
        "discussion": "github.event.discussion.author_association",
        "discussion_comment": "github.event.comment.author_association",
    }
    for event_name, association in association_fields.items():
        assert f"github.event_name == '{event_name}'" in condition
        for trusted in ("OWNER", "MEMBER", "COLLABORATOR"):
            assert f"{association} != '{trusted}'" in condition

    assert set(re.findall(r"secrets\.([A-Z][A-Z0-9_]*)", source)) == {
        "SLACK_COMMUNITY_ACTIVITY_WEBHOOK_URL"
    }
    post = activity["steps"][0]
    assert post["env"]["SLACK_WEBHOOK_URL"] == (
        "${{ secrets.SLACK_COMMUNITY_ACTIVITY_WEBHOOK_URL }}"
    )

    script = post["run"]
    assert "${{" not in script
    assert "External maintainer request" in script

    ready_script = ready["steps"][0]["run"]
    assert "for attempt in {1..30}; do" in ready_script
    assert "sleep 10" in ready_script
    for check in ("Community CPU / Required", "PR Metadata / Required", "DCO"):
        assert check in ready_script
    assert "sort_by(.started_at) | last" in ready_script
    assert "External PR ready for maintainer" in ready_script
    assert 'if [ -z "$SLACK_WEBHOOK_URL" ]; then' in script
    assert 'gsub("&"; "&amp;")' in script
    assert 'gsub("<"; "&lt;")' in script
    assert 'gsub(">"; "&gt;")' in script
    assert "($title | slack_escape)" in script
    assert '" + $title +' not in script
    assert "curl --fail-with-body --silent --show-error" in script
    assert "--header 'Content-Type: application/json'" in script
    assert '--data "$payload"' in script
    assert '"$SLACK_WEBHOOK_URL"' in script


def test_performance_preflight_requires_executable_native_entrypoints(tmp_path: Path) -> None:
    trtmc_bench = tmp_path / "trtmc-bench"
    worker = tmp_path / "trtmc-worker"
    hf_runner = tmp_path / "hf-runner.py"
    task_runner = tmp_path / "task-runner.py"
    for path in (trtmc_bench, worker, hf_runner, task_runner):
        path.write_text("#!/bin/sh\n")
    environment = perf_matrix.Environment(
        name="test",
        trtmc_bench=trtmc_bench,
        worker=worker,
        hf_runner=hf_runner,
        task_runner=task_runner,
        results_root=tmp_path / "results",
        scratch_root=tmp_path / "scratch",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=tmp_path / "runtime",
        bundle_retention="retain",
        local_files_only=True,
        timeout_seconds=1,
        references={},
    )

    with pytest.raises(perf_matrix.PerfMatrixError, match="trtmc-bench is not executable"):
        perf_matrix.preflight([], environment, require_runtime=False)

    trtmc_bench.chmod(0o755)
    assert perf_matrix.preflight([], environment, require_runtime=False) == []


def test_dev_images_install_the_pinned_requirements_without_deleted_docs() -> None:
    repository = Path(__file__).resolve().parents[2]
    for name in ("Dockerfile.dev.aarch64", "Dockerfile.dev.x86"):
        source = (repository / name).read_text(encoding="utf-8")
        assert "COPY community-ci.txt" in source
        assert "source-build.md" not in source


def test_gpu_free_unit_scope_keeps_family_python_in_physical_jobs(tmp_path: Path) -> None:
    context = RecordingContext(
        tmp_path,
        {"TRTMC_CORE_BUILD_DIR": str(tmp_path / "core-build")},
    )

    UnitTestRunner(context).premerge()

    python_command = context.calls[2][0]
    assert python_command[:3] == ["python", "-m", "pytest"]
    assert python_command[3:8] == [
        "core/builder/tests",
        "apps/benchmark/trtmc_benchmark/tests",
        "examples/models/cosmos3/dual_spark/test_cosmos3_dual_spark_source.py",
        (
            "examples/models/nemotron_voicechat/full_duplex/"
            "test_voicechat_full_duplex_source.py"
        ),
        "tools/tests",
    ]
    assert "families" not in python_command

    source = inspect.getsource(UnitTestRunner.premerge)
    assert '"all"' in source
    assert '"pytest"' in source
    assert '"not gpu and not trt"' in source
    assert '"cmake"' in source
    assert '"ctest"' in source
    assert '"--target"' not in source
    assert '"-R"' not in source

    physical = inspect.getsource(E2ERunner._run)
    assert '"not gpu and not trt"' in physical
    assert '"test_e2e.py"' in physical
    assert '"trtmc-{family}-unit-junit.xml"' in physical
    assert '"gpu or trt"' in physical
    assert '"pip"' not in physical
    assert "requirements.txt" not in physical


def test_docker_ensure_builds_the_current_dockerfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github_env = tmp_path / "github-env"
    manager = DockerImageManager(
        tmp_path,
        {"TRTMC_CI_IMAGE": "trtmc:test", "GITHUB_ENV": str(github_env)},
    )
    commands = []
    monkeypatch.setattr(manager, "_inspect", lambda _image: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        manager.commands,
        "run",
        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert manager.ensure() == "trtmc:test"
    assert commands == [["docker", "build", "--file", "Dockerfile", "--tag", "trtmc:test", "."]]
    assert github_env.read_text() == "TRTMC_CI_IMAGE=trtmc:test\n"


def test_container_does_not_chmod_the_shared_workspace() -> None:
    source = Path(inspect.getsourcefile(CiContainer)).read_text()
    assert '"chmod"' not in source


def test_trusted_container_mounts_hugging_face_caches_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cache = tmp_path / "cache"
    nested_cache = cache / "hub"
    modules_cache = tmp_path / "modules"
    runtime = tmp_path / "runtime"
    model = tmp_path / "model"
    for path in (workspace, nested_cache, modules_cache, runtime, model):
        path.mkdir(parents=True)
    container = CiContainer(
        {
            "TRTMC_CI_WORKSPACE": str(workspace),
            "TRTMC_CI_IMAGE": "trtmc:test",
            "TRTMC_CI_CONTAINER_NAME": "trtmc-test",
            "HF_HOME": str(cache),
            "HF_HUB_CACHE": str(nested_cache),
            "HUGGINGFACE_HUB_CACHE": str(cache),
            "HF_MODULES_CACHE": str(modules_cache),
            "TRTMC_RUNTIME_ROOT": str(runtime),
            "TRTMC_ALPHA_MODEL_DIR": str(model),
        }
    )

    _options, mounts = container._runtime_boundary()
    specifications = set(mounts[1::2])

    assert f"{cache}:{cache}:ro" in specifications
    assert f"{modules_cache}:{modules_cache}" in specifications
    assert f"{nested_cache}:{nested_cache}" not in specifications
    assert f"{runtime}:{runtime}" in specifications
    assert f"{model}:{model}" in specifications
    assert not any(
        specification.startswith(f"{cache}:{cache}") and not specification.endswith(":ro")
        for specification in specifications
    )


def test_workspace_local_hub_state_uses_the_workspace_mount(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    hub = tmp_path / "hub"
    workspace.mkdir()
    hub.mkdir()
    container = CiContainer(
        {
            "TRTMC_CI_WORKSPACE": str(workspace),
            "TRTMC_CI_IMAGE": "trtmc:test",
            "HF_HOME": str(workspace / ".ci/hf-home"),
            "HF_HUB_CACHE": str(hub),
            "HUGGINGFACE_HUB_CACHE": str(hub),
            "HF_MODULES_CACHE": str(workspace / ".ci/hf-modules"),
        }
    )
    (workspace / ".ci/hf-home").mkdir(parents=True)
    (workspace / ".ci/hf-modules").mkdir(parents=True)

    _options, mounts = container._runtime_boundary()

    assert mounts == ["-v", f"{hub}:{hub}:ro"]


def test_source_archive_carries_base_and_family_requirements(tmp_path: Path) -> None:
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/base.txt").write_text("build\n")
    family = tmp_path / "families/alpha"
    family.mkdir(parents=True)
    (family / "model.py").write_text("def build(request, writer): pass\n")
    (family / "requirements.txt").write_text("family-dependency\n")
    archive_path = tmp_path / "package.tar.gz"

    def write_archive(paths: tuple[str, ...]) -> None:
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in paths:
                payload = b"content\n"
                member = tarfile.TarInfo(f"package-0.1/{path}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

    write_archive(("requirements/base.txt",))
    with pytest.raises(CiError, match="dependency declarations are missing"):
        SourceArchiveValidator(CiContext(tmp_path, {})).validate([archive_path])

    write_archive(("requirements/base.txt", "families/alpha/requirements.txt"))
    SourceArchiveValidator(CiContext(tmp_path, {})).validate([archive_path])


def test_wheel_validation_requires_exact_new_payload(tmp_path: Path) -> None:
    family_names = ("alpha", "beta", "gamma")
    for family in family_names:
        root = tmp_path / "families" / family
        root.mkdir(parents=True)
        (root / "model.py").write_text("def build(request, writer): pass\n")
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tensorrt_model_connect/__init__.py", "")
        archive.writestr("trtmc_benchmark/__init__.py", "")
        archive.writestr("families/__init__.py", "")
        archive.writestr("tensorrt_model_connect/bin/trtmc", "")
        archive.writestr("tensorrt_model_connect/bin/trtmc_benchmark_worker", "")
        archive.writestr("tensorrt_model_connect/bin/trtmc_dataset_benchmark", "")
        archive.writestr("tensorrt_model_connect/bin/libtrtmc_core.so", "")
        archive.writestr("tensorrt_model_connect/bin/libtrtmc_runtime.so", "")
        archive.writestr("tensorrt_model_connect/bin/libtrtmc_backend_trt.so", "")
        archive.writestr("tensorrt_model_connect/bin/libtrtmc_byok_tvm_ffi.so", "")
        archive.writestr(
            "package-0.1.dist-info/entry_points.txt",
            "[console_scripts]\ntrtmc-bench = trtmc_benchmark.cli:main\n",
        )
        archive.writestr(
            "package-0.1.dist-info/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: package\n"
            "Version: 0.1\n"
            "Provides-Extra: cutedsl\n"
            "Provides-Extra: test\n",
        )
        archive.writestr("package-0.1.data/scripts/trtmc", "")
        archive.writestr("package-0.1.data/scripts/libtrtmc_core.so", "")
        archive.writestr("package-0.1.data/scripts/libtrtmc_runtime.so", "")
        for family in family_names:
            archive.writestr(
                f"families/{family}/model.py",
                (tmp_path / "families" / family / "model.py").read_bytes(),
            )
            archive.writestr(f"tensorrt_model_connect/bin/libtrtmc_model_{family}.so", "")

    WheelArchiveValidator(CiContext(tmp_path, {})).validate([wheel])

    corrupt = tmp_path / "corrupt.whl"
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(corrupt, "w") as output:
        for entry in source.infolist():
            payload = source.read(entry.filename)
            if entry.filename == "families/alpha/model.py":
                payload = b"def build(request, writer):\n    raise RuntimeError('corrupt')\n"
            output.writestr(entry, payload)
    with pytest.raises(CiError, match="family Python files differ from Source"):
        WheelArchiveValidator(CiContext(tmp_path, {})).validate([corrupt])

    helper = tmp_path / "families/alpha/helper.py"
    helper.write_text("VALUE = 1\n")
    with pytest.raises(CiError, match="family Python files are missing"):
        WheelArchiveValidator(CiContext(tmp_path, {})).validate([wheel])
    helper.unlink()

    requirements = tmp_path / "families/alpha/requirements.txt"
    requirements.write_text("family-dependency\n")
    with pytest.raises(CiError, match="family build data is missing"):
        WheelArchiveValidator(CiContext(tmp_path, {})).validate([wheel])
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("families/alpha/requirements.txt", "family-dependency\n")

    benchmark_asset = tmp_path / "families/alpha/tests/data/input.txt"
    benchmark_asset.parent.mkdir(parents=True)
    benchmark_asset.write_text("input\n")
    with pytest.raises(CiError, match="benchmark catalog is missing"):
        WheelArchiveValidator(CiContext(tmp_path, {})).validate([wheel])
    benchmark_asset.unlink()

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("tensorrt_model_connect/bin/libtrtmc_backend_trt_11_1.so", "")
    with pytest.raises(CiError, match="unaliased"):
        WheelArchiveValidator(CiContext(tmp_path, {})).validate([wheel])


def test_native_validation_rejects_unresolved_family_symbols(tmp_path: Path) -> None:
    def compile_library(name: str, source: str) -> None:
        source_path = tmp_path / f"{name}.c"
        source_path.write_text(source)
        subprocess.run(
            ["cc", "-shared", "-fPIC", source_path, "-o", tmp_path / name],
            check=True,
        )

    compile_library("libtrtmc_core.so", "void core_symbol(void) {}\n")
    compile_library("libtrtmc_runtime.so", "void runtime_symbol(void) {}\n")
    compile_library("libtrtmc_backend_trt.so", "void backend_symbol(void) {}\n")
    compile_library("libtrtmc_byok_tvm_ffi.so", "void byok_symbol(void) {}\n")
    compile_library("libtrtmc_model_alpha.so", "void alpha_symbol(void) {}\n")
    compile_library("libtrtmc_model_beta.so", "void beta_symbol(void) {}\n")
    load_native_libraries(tmp_path, ("alpha", "beta"))

    compile_library(
        "libtrtmc_model_beta.so",
        "extern void missing_symbol(void); void beta_symbol(void) { missing_symbol(); }\n",
    )
    with pytest.raises(CiError, match="undefined symbol: missing_symbol"):
        load_native_libraries(tmp_path, ("alpha", "beta"))
