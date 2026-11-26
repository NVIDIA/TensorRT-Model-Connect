# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One session-wide check that a test module cannot make.

``test_e2e.py`` refuses a selection that names this family and matches none of its cases. It
cannot refuse ``--e2e-testcase <typo>`` on its own, though: a testcase filter that names nobody
makes every family deselect itself, and the run then passes having tested nothing. That needs a
look at what the whole session collected, which is a collection hook - and pytest registers
collection hooks from ``conftest.py``, not from ordinary test modules.
"""

from __future__ import annotations

import fnmatch

import pytest

_FAMILY_PREFIX = "families/tipsv2/"


def _testcase_filters(config) -> set[str]:
    # Spelled out again rather than imported from test_e2e, which this file is loaded before.
    return {
        value.strip()
        for raw in config.getoption("--e2e-testcase", default=[]) or []
        for value in str(raw).split(",")
        if value.strip()
    }


def _has_model_filters(config) -> bool:
    return bool(config.getoption("--e2e-model", default=[])
                or config.getoption("--e2e-models-file", default=None))


def pytest_collection_modifyitems(session, config, items) -> None:
    """Fail a **testcase-only** selection that matches none of the session's E2E cases.

    Quiet when another collected family owns the named case. Quiet, too, as soon as a model
    selector is given: then each family's own file decides, raising when the selection names it
    and matches nothing and deselecting itself otherwise, and a session-wide rule on top of that
    would turn one family's legitimate skip into an error.
    """
    del session
    testcases = _testcase_filters(config)
    if not testcases or _has_model_filters(config):
        return
    # A family that deselected itself leaves one placeholder item whose `case_name` is pytest's
    # NOTSET sentinel rather than a case, so only real names count.
    #
    # Only E2E items count, and what makes an item one is **the test function's own name**, not
    # the file it sits in: a `test_*e2e` function is the E2E wherever a family keeps it, and a
    # unit test parametrized with a `case_name` is not one even inside `test_e2e.py`. Matching
    # the file would reject a case another family collected from a differently named file, and
    # would let an unrelated parametrization here satisfy a check that is meant to catch a
    # selection running no E2E case at all. `originalname` is the undecorated function name,
    # which a parametrized item keeps.
    e2e_items = [item for item in items
                 if fnmatch.fnmatch(getattr(item, "originalname", "") or "", "test_*e2e")]

    def case_name(item):
        spec = getattr(item, "callspec", None)
        value = spec.params.get("case_name") if spec else None
        return value if isinstance(value, str) else None

    # Every family's cases count, not only this one's: the check is whether the selection names
    # anything the **session** collected.
    collected = {name for name in map(case_name, e2e_items) if name}
    if testcases & collected:
        return

    # Nothing matched. Before calling that a typo, allow for a sibling whose cases this file
    # cannot see: a family is free to parametrize on something other than `case_name`
    # (`families/clef` and `families/laya` parametrize on the manifest and the case). If any
    # foreign E2E item is opaque that way, the selection may well be correct for the family that
    # owns it, and aborting the whole session would make this family's presence break someone
    # else's run - which is what family isolation forbids. Only when every E2E item in the
    # session is one this file can read, and none matches, is the selection a typo.
    if any(not item.nodeid.startswith(_FAMILY_PREFIX) and case_name(item) is None
           for item in e2e_items):
        return

    raise pytest.UsageError(
        f"--e2e-testcase {sorted(testcases)} matches none of the E2E cases this session "
        f"collected ({sorted(collected)})")
