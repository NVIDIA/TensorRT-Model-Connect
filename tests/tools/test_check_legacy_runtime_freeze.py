"""Unit tests for tools/check_legacy_runtime_freeze.py::is_protected.

Intent:
    Cover the pure string-matching guard that blocks accidental reintroduction
    of the deleted legacy compatibility runtime paths (src/cabi/pipeline/,
    src/cabi/registry/, src/cabi/factories/). is_protected is the heart of
    the CI freeze gate — a silent regression (e.g. stripping .strip(),
    mis-editing PROTECTED_PREFIXES) would let blocked-path changes slip
    through unnoticed.

Preconditions:
    - tools.check_legacy_runtime_freeze is importable (pure-Python module,
      no git invocation at import time).

Postconditions:
    - is_protected returns True for a representative path under each
      currently-protected prefix.
    - is_protected returns False for paths outside the protected set.
    - is_protected strips surrounding whitespace before matching, matching
      the real git-diff output contract.
    - PROTECTED_PREFIXES is referenced directly so the assertions stay in
      sync if the tuple is updated.

Trace IDs:
    ARCH-cabi-freeze, UD-legacy-runtime-guard, UT-check-legacy-runtime-freeze-is-protected.
"""

from __future__ import annotations

import pytest

from tools import check_legacy_runtime_freeze


EXPECTED_PROTECTED_PREFIXES = (
    "src/cabi/pipeline/",
    "src/cabi/registry/",
    "src/cabi/factories/",
)


def test_protected_prefixes_tuple_matches_expected() -> None:
    """Pin the guarded prefix set so assertions below stay meaningful.

    If this ever fails, either the test below or the module changed
    intentionally — update both together.
    """
    assert check_legacy_runtime_freeze.PROTECTED_PREFIXES == EXPECTED_PROTECTED_PREFIXES


@pytest.mark.parametrize("prefix", EXPECTED_PROTECTED_PREFIXES)
def test_is_protected_matches_each_prefix(prefix: str) -> None:
    """Each currently-protected prefix must match a representative file path."""
    sample = prefix + "foo.cpp"
    assert check_legacy_runtime_freeze.is_protected(sample) is True


def test_is_protected_matches_all_prefixes_from_module_tuple() -> None:
    """Iterate directly over the module tuple so renames/additions are caught."""
    for prefix in check_legacy_runtime_freeze.PROTECTED_PREFIXES:
        sample = f"{prefix}sample_file.cpp"
        assert check_legacy_runtime_freeze.is_protected(sample) is True, (
            f"expected is_protected to accept path under {prefix!r}, got False"
        )


@pytest.mark.parametrize(
    "path",
    [
        "src/runtime/models/text_generation/plugin.cpp",
        "tools/coverage_map/fetch_latest.py",
        "include/trtmc/runtime/pipeline_plugin.h",
        "README.md",
        "",
        "src/cabi/api/trtmc_c.cpp",  # cabi/api/ is NOT protected
    ],
)
def test_is_protected_rejects_non_matching_paths(path: str) -> None:
    """Paths outside the protected prefix set must not be flagged."""
    assert check_legacy_runtime_freeze.is_protected(path) is False


def test_is_protected_strips_surrounding_whitespace() -> None:
    """Confirm .strip() normalization — git diff output may carry trailing
    newlines or leading spaces, and the guard must still fire."""
    assert check_legacy_runtime_freeze.is_protected("  src/cabi/pipeline/foo.cpp\n") is True
    assert check_legacy_runtime_freeze.is_protected("\tsrc/cabi/registry/bar.cpp\t") is True
    assert check_legacy_runtime_freeze.is_protected("\nsrc/cabi/factories/baz.cpp  ") is True


def test_is_protected_does_not_match_prefix_without_trailing_slash() -> None:
    """Prefix boundaries matter — a sibling file like src/cabi/pipeline_notes.md
    must not be swept up by the src/cabi/pipeline/ prefix."""
    # The tuple entries end in '/', so a bare 'src/cabi/pipeline' (no trailing
    # slash, nothing after) should not match.
    assert check_legacy_runtime_freeze.is_protected("src/cabi/pipeline") is False
    assert check_legacy_runtime_freeze.is_protected("src/cabi/pipeline_notes.md") is False


def test_main_blocks_explicit_protected_file_list(capsys) -> None:
    rc = check_legacy_runtime_freeze.main(
        ["--files", "src/cabi/pipeline/foo.cpp", "src/runtime/models/text_generation/plugin.cpp"]
    )

    captured = capsys.readouterr()

    assert rc == 1
    assert "src/cabi/pipeline/foo.cpp" in captured.err
    assert "--allow-override" in captured.err


def test_main_allow_override_is_explicit_flag(capsys) -> None:
    rc = check_legacy_runtime_freeze.main(
        ["--allow-override", "--files", "src/cabi/pipeline/foo.cpp"]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert "override enabled via --allow-override" in captured.err
