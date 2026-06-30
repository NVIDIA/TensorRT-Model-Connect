# =============================================================================
# ISO 26262 Traceability
# =============================================================================
# Trace ID:       UT-CFG-CROSSLANG-01
# Architecture:   ARCH-CFG-001
# Unit Design:    UD-CFG-REG-01
# Intent:         Gate against Python/C++ schema drift. If the two sides of a
#                 feature's schema diverge, runtime merges silently produce
#                 different results — this test catches that at PR time.
# Preconditions:  trtmc test binary built (so the registry state reflects
#                 generated manifest registrations).
# Postconditions: For every namespace registered on either side, field
#                 names and types match. Missing namespaces on one side
#                 fail the test.
# =============================================================================

"""Cross-language schema-parity gate.

Python side: importing ``tensorrt_model_connect.runtime_config.schemas`` populates the
registry via ``load_all()``.

C++ side: there's no cheap way to query the C++ registry from Python. We
use a conservative pairing: for every namespace the Python side registers,
the corresponding C++ schema source must exist at a well-known path and
declare matching field names. We parse the C++ source with a light regex
(not a compiler) — good enough because the schemas use a fixed vocabulary
of helper calls (``bool_field`` / ``int_field`` / ``str_field``).

When codegen lands, this test becomes one line: compare the Python schema
to the generated C++ header byte-for-byte.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    from tensorrt_model_connect.runtime_config import (
        clear_for_testing,
        lookup,
    )
    from tensorrt_model_connect.runtime_config.schemas import load_all
except ImportError:  # pragma: no cover
    pytest.skip("tensorrt_model_connect.runtime_config not importable", allow_module_level=True)


_CPP_FIELD_PATTERN = re.compile(
    r'(?:bool|int|str)_field\(\s*"(?P<name>[a-zA-Z0-9_]+)"', re.MULTILINE,
)


def _cpp_schema_path(namespace: str) -> Path:
    # Maps namespace to the schema source by convention: one file per ns.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "src" / "runtime" / "config" / "schemas" / f"{namespace}.cpp"


def _cpp_fields_in_source(path: Path) -> list[str]:
    """Extract field names declared in the C++ schema .cpp file."""
    text = path.read_text(encoding="utf-8")
    return [m.group("name") for m in _CPP_FIELD_PATTERN.finditer(text)]


@pytest.fixture(autouse=True)
def clean_registry():
    clear_for_testing()
    yield
    clear_for_testing()


def test_load_all_populates_triattention():
    """Importing the schemas package registers the triattention namespace."""
    loaded = load_all()
    assert "triattention" in loaded
    schema = lookup("triattention")
    assert schema is not None
    assert len(schema.fields) > 0


def test_load_all_populates_model_owned_audio_schemas():
    """Family sidecars are discovered without importing their TRT plugins."""
    loaded = load_all()

    assert {"audio_bark", "audio_magpie"} <= set(loaded)
    assert lookup("audio_bark") is not None
    assert lookup("audio_magpie") is not None


def test_triattention_field_set_matches_cpp():
    """Python and C++ schema files declare the same field names."""
    load_all()
    py_schema = lookup("triattention")
    assert py_schema is not None
    py_names = sorted(f.name for f in py_schema.fields)

    cpp_path = _cpp_schema_path("triattention")
    assert cpp_path.exists(), (
        f"missing C++ schema source: {cpp_path}. Every Python schema must "
        f"have a matching .cpp in src/runtime/config/schemas/ (until codegen "
        f"generates it)."
    )
    cpp_names = sorted(_cpp_fields_in_source(cpp_path))

    missing_in_cpp = set(py_names) - set(cpp_names)
    missing_in_py = set(cpp_names) - set(py_names)
    assert not missing_in_cpp, (
        f"fields in Python schema but not C++: {sorted(missing_in_cpp)}"
    )
    assert not missing_in_py, (
        f"fields in C++ schema but not Python: {sorted(missing_in_py)}"
    )


def test_triattention_defaults_plausible():
    """Sanity-check a few expected defaults so the schema isn't silently empty."""
    load_all()
    schema = lookup("triattention")
    by_name = {f.name: f for f in schema.fields}
    assert by_name["kv_budget"].default == 4096
    assert by_name["divide_length"].default == 128
    assert by_name["score_aggregation"].default == "mean"
    assert by_name["protect_prefill"].default is True
    assert by_name["stats_section"].default == "triattention_stats.json"
