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

    def names(item):
        """Every name this item could be selected by, from what pytest exposes publicly.

        Two sources, because a family is free to parametrize however it likes: the `case_name`
        parameter, which this family uses, and the item's **parameter id**, which is the last
        bracketed part of the nodeid. `families/clef` and `families/laya` parametrize on
        `(manifest, case)` and pass `id=case["name"]`, so the name a caller would type is right
        there even though `case_name` is not.

        A family whose items carry the name in neither place cannot be selected by name by
        anyone, so there is nothing to defer to.
        """
        spec = getattr(item, "callspec", None)
        value = spec.params.get("case_name") if spec else None
        if isinstance(value, str):
            yield value
        nodeid = getattr(item, "nodeid", "") or ""
        if nodeid.endswith("]") and "[" in nodeid:
            identifier = nodeid[nodeid.rindex("[") + 1:-1]
            # A family that deselected itself leaves one placeholder whose id is pytest's NOTSET
            # sentinel rather than a case; it is not a name anyone can select.
            if identifier != "NOTSET":
                yield identifier

    # Every family's names count, this one's and its siblings'. The question is only whether the
    # **session** collected the name that was asked for - which is decidable from what pytest
    # already publishes, without this file reaching into how another family parametrizes.
    collected = {name for item in e2e_items for name in names(item)}
    if testcases & collected:
        return

    raise pytest.UsageError(
        f"--e2e-testcase {sorted(testcases)} matches none of the E2E cases this session "
        f"collected ({sorted(collected)})")
