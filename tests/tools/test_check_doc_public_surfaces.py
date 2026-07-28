# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public-surface documentation coverage gate."""

from __future__ import annotations

from pathlib import Path

from tools import check_doc_public_surfaces as surfaces


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cpp_long_options_require_complete_string_literals() -> None:
    source = '''
    if (arg == "--kv-cache-size" || arg == "--kv_cache_size") {}
    if (arg.rfind("--kv-cache-size=", 0) == 0) {}
    const char* error = "--kv-cache-size expects a value";
    '''

    assert surfaces.extract_cpp_long_options(source) == {
        "--kv-cache-size",
        "--kv_cache_size",
    }
    assert "--kv" not in surfaces.extract_cpp_long_options(source)


def test_python_function_parameters_include_keyword_only_arguments() -> None:
    source = "def build(model, output, *, precision=None, verbose=False):\n    pass\n"

    assert surfaces.extract_python_function_parameters(source, "build") == {
        "model",
        "output",
        "precision",
        "verbose",
    }


def test_argparse_extractor_ignores_short_and_nonliteral_options() -> None:
    source = '''
def build_parser():
    parser = object()
    parser.add_argument("-o", "--output")
    parser.add_argument("--worker")
    parser.add_argument(option_name)
    return parser
'''

    assert surfaces.extract_argparse_long_options(source, "build_parser") == {
        "--output",
        "--worker",
    }


def test_schema_extractors_return_namespace_field_contracts() -> None:
    python_source = '''
SCHEMA = Schema(
    namespace="demo",
    fields=(
        ConfigField(name="enabled", type_tag="bool", default=helper("not_a_field")),
        int_field("limit", 1),
    ),
)
'''
    cpp_source = '''
Schema make_demo_schema() {
    return Schema{
        "demo",
        {
            ConfigField{"enabled", "bool", {}, {}, nullptr},
            ConfigField{"limit", "int32", {}, {}, nullptr},
        },
    };
}
'''

    expected = {"demo": {"enabled", "limit"}}
    assert surfaces.extract_python_schemas(python_source) == expected
    assert surfaces.extract_cpp_schemas(cpp_source) == expected


def test_cpp_public_method_extractors_ignore_special_and_nested_methods() -> None:
    source = '''
class Demo {
  public:
    virtual ~Demo() = default;
    virtual bool supports() const { return true; }
    virtual std::vector<int>
    values() const { return {}; }
};
class Pool {
  public:
    class Lease {
      public:
        void nested();
    };
    Pool();
    ~Pool();
    Lease acquire();
    bool supports() const;
  private:
    void hidden();
};
'''

    assert surfaces.extract_cpp_virtual_methods(source, "Demo") == {
        "supports",
        "values",
    }
    assert surfaces.extract_cpp_top_level_public_methods(source, "Pool") == {
        "acquire",
        "supports",
    }


def test_mapping_passes_with_documented_tokens_and_reasoned_alias(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "docs/reference.md"
    doc.parent.mkdir()
    doc.write_text("Use `--canonical` and `build_option`.\n", encoding="utf-8")
    public = {"demo": {"--canonical", "--compat_alias", "build_option"}}
    mapping = {
        "surfaces": {
            "demo": {
                "documents": ["docs/reference.md"],
                "allowlist": {
                    "--compat_alias": {
                        "canonical": "--canonical",
                        "reason": "Compatibility spelling retained for existing scripts.",
                    }
                },
            }
        }
    }

    report = surfaces.check_mappings(tmp_path, public, mapping)

    assert report.ok
    assert report.allowlisted_count == 1


def test_mapping_fails_when_a_public_token_is_undocumented(tmp_path: Path) -> None:
    doc = tmp_path / "docs/reference.md"
    doc.parent.mkdir()
    doc.write_text("Use `--present`.\n", encoding="utf-8")
    public = {"demo": {"--present", "--missing"}}
    mapping = {
        "surfaces": {
            "demo": {
                "documents": ["docs/reference.md"],
                "allowlist": {},
            }
        }
    }

    report = surfaces.check_mappings(tmp_path, public, mapping)

    assert not report.ok
    assert any(finding.token == "--missing" for finding in report.findings)


def test_mapping_rejects_stale_or_unreasoned_allowlist_entries(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "docs/reference.md"
    doc.parent.mkdir()
    doc.write_text("Reference.\n", encoding="utf-8")
    public = {"demo": {"current"}}
    mapping = {
        "surfaces": {
            "demo": {
                "documents": ["docs/reference.md"],
                "allowlist": {
                    "current": {"reason": "short"},
                    "retired": {"reason": "This token no longer exists in source."},
                },
            }
        }
    }

    report = surfaces.check_mappings(tmp_path, public, mapping)

    assert not report.ok
    assert {finding.token for finding in report.findings} == {"current", "retired"}


def test_repository_public_surfaces_are_documented() -> None:
    mapping = surfaces.load_mapping(REPO_ROOT / surfaces.DEFAULT_MAPPING_PATH)
    public = surfaces.collect_public_surfaces(REPO_ROOT)

    report = surfaces.check_mappings(REPO_ROOT, public, mapping)

    assert report.ok, "\n".join(str(finding) for finding in report.findings)
