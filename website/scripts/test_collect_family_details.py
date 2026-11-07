# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial checks for source extraction and CPU evidence boundaries."""

import argparse
import ast
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).with_name("collect-family-details.py")
SPEC = importlib.util.spec_from_file_location("collect_family_details", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class ConfigurationExtractionTests(unittest.TestCase):
    def test_keeps_owner_defaults_class_variables_and_properties_distinct(self):
        text = '''
@dataclass
class ModelConfig:
    """Checkpoint dimensions."""
    hidden_size: int
    # Fixed compatibility value, not a user-selectable option.
    max_batch_size: ClassVar[int] = 1
    widths: list[int] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // 8
'''
        node = ast.parse(text).body[0]
        result = collector.class_details(Path("/repo"), Path("/repo/families/demo/config.py"), node, text.splitlines())
        fields = {field["name"]: field for field in result["fields"]}
        self.assertTrue(fields["hidden_size"]["required"])
        self.assertIsNone(fields["hidden_size"]["default"])
        self.assertTrue(fields["max_batch_size"]["readOnly"])
        self.assertIn("not a user-selectable", fields["max_batch_size"]["description"])
        self.assertEqual(fields["widths"]["default"], "field(default_factory=list)")
        self.assertTrue(fields["head_dim"]["readOnly"])
        self.assertEqual(fields["head_dim"]["source"]["line"], 11)

    def test_keeps_dictionary_configs_and_lookup_expressions_without_guessing_types(self):
        text = '''
def resolve(raw):
    return {"width": int(raw.get("hidden_size", 768)), "encoder": {"depth": raw.get("depth", 12)}}
'''
        result = collector.dictionary_details(Path("/repo"), Path("/repo/families/demo/config.py"), ast.parse(text))
        fields = result[0]["fields"]
        width = next(field for field in fields if field["name"] == "width")
        self.assertEqual(width["type"], "derived")
        self.assertIn("raw.get", width["default"])
        hidden = next(field for field in fields if field["name"] == "hidden_size")
        self.assertEqual(hidden["type"], "checkpoint metadata")
        self.assertEqual(hidden["default"], "768")
        self.assertTrue(hidden["readOnly"])
        self.assertIn("not a CLI option", hidden["description"])

    def test_handler_signature_mismatch_is_an_issue_not_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            family = root / "families/demo"
            family.mkdir(parents=True)
            (family / "cli.py").write_text("def build(*, model, output, required_capacity):\n    pass\n")
            result = collector.inspect_handler(root, family, {
                "executor": "python", "handler": "cli:build",
                "arguments": [{"name": "model"}, {"name": "output"}, {"name": "unknown"}],
            })
            self.assertEqual(result["handlerStatus"], "issues")
            self.assertEqual(len(result["issues"]), 2)
            self.assertIn("unknown", result["issues"][0])
            self.assertIn("required_capacity", result["issues"][1])

    def test_dictionary_branches_keep_conflicting_defaults_and_source(self):
        text = '''
def resolve(raw):
    theta = raw.get("rope_theta", None)
    if theta is None:
        theta = raw.get("rope_theta", 10000.0)
    return {"text": {"width": 64}, "vision": {"width": 256}}
'''
        result = collector.dictionary_details(Path("/repo"), Path("/repo/families/demo/config.py"), ast.parse(text))[0]
        lookups = [field for field in result["fields"] if field["name"] == "rope_theta"]
        self.assertEqual([field["default"] for field in lookups], ["None", "10000.0"])
        self.assertNotEqual(lookups[0]["source"]["line"], lookups[1]["source"]["line"])
        widths = [field["default"] for field in result["fields"] if field["name"] == "width"]
        self.assertEqual(widths, ["64", "256"])
        self.assertIn("if theta is None", result["declaration"])

    def test_missing_cli_keeps_shared_command_boundary_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = collector.cli_details(root, root / "demo", None)
            self.assertEqual(result["status"], "missing")
            self.assertIn("shared build command remains available", result["note"])
            self.assertEqual(result["fallbackSource"]["path"], "core/builder/tensorrt_model_connect/build_cli.py")


class GuardReachabilityTests(unittest.TestCase):
    def guards(self, body, **inputs):
        return collector.direct_build_guards(ast.parse(body), SimpleNamespace(**inputs))

    def test_direct_rejection_is_reported(self):
        issues, checked, _ = self.guards('''
def build(request, writer):
    if request.max_sequence_length is not None:
        raise NotImplementedError("unsupported sequence capacity")
''', max_sequence_length=32)
        self.assertEqual(checked, 1)
        self.assertEqual(len(issues), 1)
        self.assertIn("unsupported sequence capacity", issues[0])

    def test_early_family_dispatch_prevents_unreachable_false_positive(self):
        issues, checked, _ = self.guards('''
def build(request, writer):
    if request.task == "embedding":
        return build_embedding(request, writer)
    if request.task != "text_generation":
        raise ValueError("text generation only")
''', task="embedding")
        self.assertEqual(issues, [])
        self.assertEqual(checked, 1)

    def test_unknown_early_return_defers_later_guard(self):
        issues, _, deferred = self.guards('''
def build(request, writer):
    if checkpoint_dispatch(request):
        return custom_build(request, writer)
    if request.task != "text_generation":
        raise ValueError("text generation only")
''', task="embedding")
        self.assertEqual(issues, [])
        self.assertEqual(deferred, 1)

    def test_guard_evaluation_never_calls_model_code(self):
        issues, checked, deferred = self.guards('''
def build(request, writer):
    if dangerous_checkpoint_loader(request.model_dir):
        raise ValueError("checkpoint dependent")
    if request.precision != "bf16":
        raise ValueError("bf16 required")
''', model_dir="not-a-real-path", precision="bf16")
        self.assertEqual(issues, [])
        self.assertEqual(checked, 1)
        self.assertEqual(deferred, 1)

    def test_else_and_unconditional_rejections_are_reported(self):
        issues, _, _ = self.guards('''
def build(request, writer):
    if request.task == "encoding":
        pass
    else:
        raise ValueError("encoding required")
''', task="bad")
        self.assertIn("encoding required", issues[0])
        issues, _, _ = self.guards('''
def build(request, writer):
    raise NotImplementedError("builder not implemented")
''')
        self.assertIn("builder not implemented", issues[0])

    def test_nested_return_only_short_circuits_when_its_condition_matches(self):
        issues, _, _ = self.guards('''
def build(request, writer):
    if request.task == "encoding":
        if request.precision == "bf16":
            return supported_build(request, writer)
        raise ValueError("encoding requires bf16")
''', task="encoding", precision="fp16")
        self.assertIn("encoding requires bf16", issues[0])

    def test_request_normalization_cannot_create_a_false_rejection(self):
        issues, checked, deferred = self.guards('''
def build(request, writer):
    request = normalize_request(request)
    if request.precision != "bf16":
        raise ValueError("bf16 required")
''', precision="fp16")
        self.assertEqual(issues, [])
        self.assertEqual(checked, 0)
        self.assertEqual(deferred, 1)


class ProfileEvidenceTests(unittest.TestCase):
    def check(self, extra=None, precision="fp32", settings=None, flags=(), coerce=False):
        @dataclass
        class Request:
            model_dir: Path
            output_path: Path
            family: str
            task: str
            precision: str
            max_sequence_length: int | None = None

        @dataclass
        class SharedRequest(Request):
            context_parallel_size: int = 1

        def coerce_request(request):
            if request.context_parallel_size != 1:
                raise ValueError("family does not support context parallelism")
            return request

        raw = {"name": "fixture", "bundle": "fixture.bundle", "family": "demo", "task": "encoding",
               "precision": precision, "testcases": [{"name": "fixture"}], **(extra or {})}
        model = SimpleNamespace(build_settings=settings or {})
        catalog = SimpleNamespace(
            _MANIFEST_METADATA_FIELDS={"name", "bundle", "family", "task", "precision", "testcases", "context_parallel_size"},
            ManifestCatalog=lambda root: SimpleNamespace(_load=lambda path: model),
        )

        def parser():
            result = argparse.ArgumentParser()
            result.add_argument("command")
            result.add_argument("model")
            result.add_argument("--precision", choices=["fp32", "fp16"])
            result.add_argument("--task")
            return result

        modules = {"catalog": catalog, "family_cli": None, "build_cli": SimpleNamespace(_parser=parser), "shared_request": SharedRequest,
                   "builder": SimpleNamespace(_build_command=lambda *args: (
                       "python3", "-m", "builder", "build", "checkpoint", "--precision", precision, "--task", "encoding", *flags))}
        with tempfile.TemporaryDirectory() as temporary:
            family = Path(temporary) / "families/demo"
            family.mkdir(parents=True)
            path = family / "fixture.json"
            path.write_text(json.dumps(raw))
            return collector.profile_check(path, family, modules, Request, coerce_request if coerce else None, {})

    def test_cli_failure_does_not_hide_independent_python_request_check(self):
        result = self.check(precision="mixed_bf16_fp32")
        self.assertEqual(result["status"], "issues")
        statuses = {check["name"]: check["status"] for check in result["checks"]}
        self.assertEqual(statuses["CLI parsing"], "issues")
        self.assertEqual(statuses["BuildRequest"], "checked")
        self.assertIn("TensorRT build", result["boundary"])

    def test_unknown_manifest_field_is_not_silently_claimed_checked(self):
        result = self.check(extra={"unowned_knob": 4})
        self.assertEqual(result["status"], "issues")
        self.assertIn("unowned_knob", result["issues"][0])
        self.assertIn("ignored", result["issues"][0])

    def test_serializer_dropping_known_build_setting_is_reported(self):
        result = self.check(extra={"max_sequence_length": 42}, settings={"max_sequence_length": 42})
        self.assertEqual(result["status"], "issues")
        self.assertIn("not preserved", result["issues"][0])
        self.assertIn("max_sequence_length", result["issues"][0])

    def test_metadata_parallelism_cannot_disappear_before_serialization(self):
        result = self.check(extra={"context_parallel_size": 4}, coerce=True)
        self.assertEqual(result["status"], "issues")
        self.assertTrue(any("not represented" in issue for issue in result["issues"]))
        self.assertTrue(any("does not support context parallelism" in issue for issue in result["issues"]))
        baseline = self.check(extra={"context_parallel_size": 1}, coerce=True)
        self.assertEqual(baseline["status"], "checked")

    def test_missing_case_adapter_does_not_invent_a_model_failure(self):
        result = self.check()
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["caseChecks"][0]["status"], "not-checked")


if __name__ == "__main__":
    unittest.main()
