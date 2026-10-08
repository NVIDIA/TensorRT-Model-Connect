<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Advisory documentation review

Maintainers can select **Actions > Documentation Check > Run workflow** on
`main` to review README, Quick Start, and Build from Source at one immutable
commit. Each document is compared with the other two as references. The LLM
review covers concrete wording and contradictory instructions, not execution,
model accuracy, performance, or proof that links work.

The public workflow only authorizes and dispatches the request. The optional
SDK, inference credential, network access, and reports belong to the protected
CI environment. Results are private Actions artifacts (`report.json` and
`summary.md`), never GitHub Pages content. A successful dispatch is not proof
that the review completed. Consult the private run for the actual result.

The adapter and its local invocation are documented in the companion CI
repository's operator guide. It reads Source documents as data and never
executes Source code with the inference credential.

Exit codes are `0` (completed, passed), `2` (completed with advisory findings),
and `1` (setup, transport, or malformed-result error). CI accepts only `0` and
`2`; errors fail visibly. Human reviewers decide whether findings are valid.
This manually triggered review is not a required merge gate and does not
change existing test criteria. It does not upload results to a dashboard.

Rollout requires the companion private workflow and its inference credential
before using the public entry point. No optional SDK is needed by Source CPU
tests. The companion controller tests exercise the adapter with a fake client.
