"""Tests for tools/coverage_map/ -- coverage-based test selection.

Trace: ARCH-CI-001, UD-CI-COVERAGE
Intent: Validate coverage-map test selection pipeline including Python/C++ collection, merge, and selection
Preconditions: Synthetic coverage databases and gcovr JSON files are created in temp directories
Postconditions: Coverage maps are correctly merged and test selection returns expected test sets for changed files
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from coverage_map.python_collector import parse_coverage_db  # noqa: E402
from coverage_map.cpp_collector import parse_gcovr_json, build_cpp_map_from_jsons  # noqa: E402
from coverage_map.generate import merge_maps, validate_map, load_coverage_map, main as generate_main  # noqa: E402
from coverage_map.select_tests import select_tests  # noqa: E402
from coverage_map.fetch_latest import resolve_coverage_map  # noqa: E402


def _create_fake_coverage_db(db_path: Path, data: dict) -> None:
    """Create a minimal coverage.py SQLite database.

    data: {source_path: {context_name: [line_numbers]}}
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS coverage_schema (version INTEGER)")
    conn.execute("INSERT INTO coverage_schema VALUES (7)")
    conn.execute("CREATE TABLE IF NOT EXISTS file (id INTEGER PRIMARY KEY, path TEXT UNIQUE)")
    conn.execute("CREATE TABLE IF NOT EXISTS context (id INTEGER PRIMARY KEY, context TEXT UNIQUE)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS line_bits "
        "(file_id INTEGER, context_id INTEGER, numbits BLOB)"
    )
    file_id = 0
    ctx_id = 0
    for src_path, contexts in data.items():
        file_id += 1
        conn.execute("INSERT INTO file VALUES (?, ?)", (file_id, src_path))
        for ctx_name, _lines in contexts.items():
            ctx_id += 1
            conn.execute("INSERT INTO context VALUES (?, ?)", (ctx_id, ctx_name))
            conn.execute(
                "INSERT INTO line_bits VALUES (?, ?, ?)",
                (file_id, ctx_id, b"\x01"),
            )
    conn.commit()
    conn.close()


class TestPythonCollector:
    def test_parse_coverage_db_basic(self, tmp_path):
        """Parses a fake .coverage DB and returns source -> test mapping."""
        db_path = tmp_path / ".coverage"
        _create_fake_coverage_db(db_path, {
            "tensorrt_model_connect/tensorrt_model_connect/config.py": {
                "tests/builder/test_config.py::TestModelConfig::test_parse|run": [1, 2, 3],
                "tests/builder/test_config.py::TestModelConfig::test_vl|run": [1, 5],
            },
            "tensorrt_model_connect/tensorrt_model_connect/graph_ops.py": {
                "tests/builder/test_graph_ops.py::TestRoPE::test_basic|run": [10, 20],
            },
        })
        result = parse_coverage_db(db_path)
        assert "tensorrt_model_connect/tensorrt_model_connect/config.py" in result
        assert sorted(result["tensorrt_model_connect/tensorrt_model_connect/config.py"]) == [
            "tests/builder/test_config.py::TestModelConfig::test_parse",
            "tests/builder/test_config.py::TestModelConfig::test_vl",
        ]
        assert result["tensorrt_model_connect/tensorrt_model_connect/graph_ops.py"] == [
            "tests/builder/test_graph_ops.py::TestRoPE::test_basic",
        ]

    def test_parse_coverage_db_strips_phase(self, tmp_path):
        """Strips |run, |setup, |teardown from context names."""
        db_path = tmp_path / ".coverage"
        _create_fake_coverage_db(db_path, {
            "tensorrt_model_connect/tensorrt_model_connect/config.py": {
                "tests/builder/test_config.py::test_a|setup": [1],
                "tests/builder/test_config.py::test_a|run": [2],
                "tests/builder/test_config.py::test_a|teardown": [3],
            },
        })
        result = parse_coverage_db(db_path)
        assert result["tensorrt_model_connect/tensorrt_model_connect/config.py"] == [
            "tests/builder/test_config.py::test_a",
        ]

    def test_parse_coverage_db_empty(self, tmp_path):
        """Empty DB returns empty mapping."""
        db_path = tmp_path / ".coverage"
        _create_fake_coverage_db(db_path, {})
        result = parse_coverage_db(db_path)
        assert result == {}

    def test_parse_coverage_db_missing_file(self, tmp_path):
        """Missing DB file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_coverage_db(tmp_path / "nonexistent.db")


class TestCppCollector:
    def test_parse_gcovr_json_basic(self, tmp_path):
        """Extracts covered source files from a gcovr JSON report."""
        gcovr_data = {
            "files": [
                {
                    "filename": "/workspace/tensorrt-model-connect/src/tokenizer/vocab_tokenizer.cpp",
                    "line_covered": 20,
                    "line_total": 50,
                },
                {
                    "filename": "/workspace/tensorrt-model-connect/src/bundle/bundle_format.cpp",
                    "line_covered": 0,
                    "line_total": 30,
                },
                {
                    "filename": "/workspace/tensorrt-model-connect/include/trtmc/runtime/pipeline_plugin.h",
                    "line_covered": 5,
                    "line_total": 10,
                },
            ]
        }
        json_path = tmp_path / "cov.json"
        json_path.write_text(json.dumps(gcovr_data))
        result = parse_gcovr_json(json_path, repo_root=Path("/workspace/tensorrt-model-connect"))
        # Only files with line_covered > 0
        assert "src/tokenizer/vocab_tokenizer.cpp" in result
        assert "include/trtmc/runtime/pipeline_plugin.h" in result
        assert "src/bundle/bundle_format.cpp" not in result

    def test_parse_gcovr_json_empty(self, tmp_path):
        """Empty gcovr report returns empty set."""
        json_path = tmp_path / "cov.json"
        json_path.write_text(json.dumps({"files": []}))
        result = parse_gcovr_json(json_path, repo_root=Path("/workspace/repo"))
        assert result == set()

    def test_build_cpp_map(self, tmp_path):
        """Builds source->tests mapping from per-test gcovr JSONs."""
        (tmp_path / "test_vocab_tokenizer.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/tokenizer/vocab_tokenizer.cpp", "line_covered": 20, "line_total": 50},
            {"filename": "/repo/src/utils/text_parsers.cpp", "line_covered": 5, "line_total": 30},
        ]}))
        (tmp_path / "test_bundle_format.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/bundle/bundle_format.cpp", "line_covered": 10, "line_total": 20},
        ]}))
        result = build_cpp_map_from_jsons(tmp_path, repo_root=Path("/repo"))
        assert result["src/tokenizer/vocab_tokenizer.cpp"] == ["test_vocab_tokenizer"]
        assert result["src/utils/text_parsers.cpp"] == ["test_vocab_tokenizer"]
        assert result["src/bundle/bundle_format.cpp"] == ["test_bundle_format"]

    def test_build_cpp_map_shared_source(self, tmp_path):
        """Two tests covering the same source file are both listed."""
        (tmp_path / "test_a.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/common.cpp", "line_covered": 5, "line_total": 10},
        ]}))
        (tmp_path / "test_b.json").write_text(json.dumps({"files": [
            {"filename": "/repo/src/common.cpp", "line_covered": 3, "line_total": 10},
        ]}))
        result = build_cpp_map_from_jsons(tmp_path, repo_root=Path("/repo"))
        assert sorted(result["src/common.cpp"]) == ["test_a", "test_b"]


class TestGenerate:
    def test_merge_maps_disjoint(self):
        """Merging two disjoint maps produces their union."""
        py_map = {"tensorrt_model_connect/config.py": ["test_config.py::test_a"]}
        cpp_map = {"src/vocab.cpp": ["test_vocab"]}
        merged = merge_maps(py_map, cpp_map)
        assert merged["tensorrt_model_connect/config.py"] == ["test_config.py::test_a"]
        assert merged["src/vocab.cpp"] == ["test_vocab"]

    def test_merge_maps_overlapping(self):
        """Overlapping keys produce union of test lists, sorted and deduplicated."""
        map_a = {"shared.py": ["test_a", "test_b"]}
        map_b = {"shared.py": ["test_b", "test_c"]}
        merged = merge_maps(map_a, map_b)
        assert merged["shared.py"] == ["test_a", "test_b", "test_c"]

    def test_validate_map_clean(self, tmp_path):
        """No warnings when all files and tests exist."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.cpp").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text("")
        coverage_map = {"src/foo.cpp": ["tests/test_foo.py::test_a"]}
        warnings = validate_map(coverage_map, tmp_path)
        assert warnings == []

    def test_validate_map_missing_source(self, tmp_path):
        """Warning when a source file in the map no longer exists."""
        coverage_map = {"src/deleted.cpp": ["test_x"]}
        warnings = validate_map(coverage_map, tmp_path)
        assert any("src/deleted.cpp" in w for w in warnings)

    def test_load_coverage_map(self, tmp_path):
        """Loads and returns the source_to_tests portion of a coverage_map.json."""
        data = {
            "meta": {"commit": "abc", "generated_at": "2026-01-01T00:00:00Z",
                     "python_tests": 10, "cpp_tests": 5},
            "source_to_tests": {"src/a.cpp": ["test_a"]},
        }
        path = tmp_path / "coverage_map.json"
        path.write_text(json.dumps(data))
        result = load_coverage_map(path)
        assert result == {"src/a.cpp": ["test_a"]}

    def test_load_coverage_map_missing_file(self, tmp_path):
        """Returns None when the map file doesn't exist."""
        result = load_coverage_map(tmp_path / "missing.json")
        assert result is None

    def test_validate_cli_does_not_require_output(self, tmp_path, monkeypatch):
        """Validation mode accepts an existing map without --output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("")
        path = tmp_path / "coverage_map.json"
        path.write_text(json.dumps({
            "meta": {},
            "source_to_tests": {"src/foo.py": ["tests/test_foo.py::test_a"]},
        }))
        monkeypatch.setattr(sys, "argv", [
            "generate.py",
            "--repo-root",
            str(tmp_path),
            "--validate",
            str(path),
        ])

        assert generate_main() == 0


class TestSelectTests:
    @pytest.fixture
    def sample_map(self):
        return {
            "src/tokenizer/vocab_tokenizer.cpp": ["test_vocab_tokenizer"],
            "src/bundle/bundle_format.cpp": ["test_bundle_format", "test_bundle_e2e"],
            "src/utils/text_parsers.cpp": ["test_text_parsers", "test_vocab_tokenizer"],
            "tensorrt_model_connect/tensorrt_model_connect/config.py": [
                "tests/builder/test_config.py::TestModelConfig::test_parse",
                "tests/builder/test_config.py::TestModelConfig::test_vl",
            ],
            "tensorrt_model_connect/tensorrt_model_connect/graph_ops.py": [
                "tests/builder/test_graph_ops.py::TestRoPE::test_basic",
            ],
        }

    def test_known_cpp_file(self, sample_map):
        """Known C++ source file returns its mapped tests."""
        result = select_tests(["src/tokenizer/vocab_tokenizer.cpp"], sample_map)
        assert sorted(result.cpp_tests) == ["test_vocab_tokenizer"]
        assert result.builder_tests == []
        assert result.fallback_tiers == []

    def test_known_python_file(self, sample_map):
        """Known Python source file returns its mapped tests."""
        result = select_tests(["tensorrt_model_connect/tensorrt_model_connect/config.py"], sample_map)
        assert sorted(result.builder_tests) == [
            "tests/builder/test_config.py::TestModelConfig::test_parse",
            "tests/builder/test_config.py::TestModelConfig::test_vl",
        ]
        assert result.cpp_tests == []

    def test_unknown_cpp_file_fallback(self, sample_map):
        """Unknown src/ file triggers cpp tier fallback."""
        result = select_tests(["src/new_module.cpp"], sample_map)
        assert "cpp" in result.fallback_tiers
        assert result.cpp_tests == []

    def test_unknown_python_file_fallback(self, sample_map):
        """Unknown tensorrt_model_connect/ file triggers builder tier fallback."""
        result = select_tests(["tensorrt_model_connect/tensorrt_model_connect/new_module.py"], sample_map)
        assert "builder" in result.fallback_tiers
        assert result.builder_tests == []

    def test_direct_python_test_file_runs_directly_without_fallback(self, sample_map):
        """Changed Python tests should run directly without forcing full-tier fallback."""
        result = select_tests([
            "tests/builder/test_family_timm_vit.py",
            "tests/tools/test_test_impact.py",
            "tests/e2e_harness/test_orchestrator_phases.py",
        ], sample_map)

        assert result.builder_tests == ["tests/builder/test_family_timm_vit.py"]
        assert result.tools_tests == [
            "tests/e2e_harness/test_orchestrator_phases.py",
            "tests/tools/test_test_impact.py",
        ]
        assert result.fallback_tiers == []

    def test_multiple_files_union(self, sample_map):
        """Multiple changed files produce union of tests."""
        result = select_tests([
            "src/tokenizer/vocab_tokenizer.cpp",
            "src/bundle/bundle_format.cpp",
        ], sample_map)
        assert sorted(result.cpp_tests) == [
            "test_bundle_e2e", "test_bundle_format", "test_vocab_tokenizer",
        ]

    def test_non_code_file_no_impact(self, sample_map):
        """docs/ file has no test impact."""
        result = select_tests(["docs/README.md"], sample_map)
        assert result.cpp_tests == []
        assert result.builder_tests == []
        assert result.fallback_tiers == []

    def test_include_header_fallback(self, sample_map):
        """Unknown include/ header triggers cpp fallback."""
        result = select_tests(["include/trtmc/new_header.h"], sample_map)
        assert "cpp" in result.fallback_tiers

    def test_mixed_cpp_and_python(self, sample_map):
        """Changes in both languages select tests from both."""
        result = select_tests([
            "src/tokenizer/vocab_tokenizer.cpp",
            "tensorrt_model_connect/tensorrt_model_connect/config.py",
        ], sample_map)
        assert "test_vocab_tokenizer" in result.cpp_tests
        assert "tests/builder/test_config.py::TestModelConfig::test_parse" in result.builder_tests

    def test_empty_changed_files(self, sample_map):
        """No changed files -> no tests."""
        result = select_tests([], sample_map)
        assert result.cpp_tests == []
        assert result.builder_tests == []
        assert result.fallback_tiers == []


class TestFetchLatest:
    def test_local_path_exists(self, tmp_path):
        """Returns True and copies when local fallback file exists."""
        path = tmp_path / "coverage_map.json"
        path.write_text('{"meta": {}, "source_to_tests": {}}')
        result = resolve_coverage_map(
            output_path=tmp_path / "output.json",
            local_fallback=str(path),
            artifact_url=None,
        )
        assert result is True
        assert (tmp_path / "output.json").exists()

    def test_local_path_missing(self, tmp_path):
        """Returns False when local fallback doesn't exist and no API."""
        result = resolve_coverage_map(
            output_path=tmp_path / "output.json",
            local_fallback=str(tmp_path / "missing.json"),
            artifact_url=None,
        )
        assert result is False

    def test_copies_to_output(self, tmp_path):
        """Copies the source file to output_path."""
        source = tmp_path / "source.json"
        source.write_text('{"meta": {}, "source_to_tests": {"a.py": ["test_a"]}}')
        output = tmp_path / "subdir" / "output.json"
        result = resolve_coverage_map(
            output_path=output,
            local_fallback=str(source),
            artifact_url=None,
        )
        assert result is True
        data = json.loads(output.read_text())
        assert "a.py" in data["source_to_tests"]
