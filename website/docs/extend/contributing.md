---
title: Contributing
---

Keep a change inside one family whenever possible. A family change must not
add a dependency on a sibling family or a central model registry.

Before opening a pull request, run the ownership validator, core tests, the
affected family tests, and the affected native target. Sign off every commit
with `git commit --signoff`.

Open the pull request as a draft and self-review its exact current head before
marking it ready. Codex users can run:

```text
$review-trtmc-pr review this draft PR as a contributor self-review
```

A documented manual review or another tool is also accepted. Record the method,
full reviewed head SHA, result, corrected findings, and unresolved risks in the
pull-request template. Any new code commit requires another self-review. For an
earlier local pass, ask `$review-trtmc-pr` to review the current branch and
working tree against `upstream/main`.

See the repository-level `CONTRIBUTING.md` for the GitHub and CI flow.
