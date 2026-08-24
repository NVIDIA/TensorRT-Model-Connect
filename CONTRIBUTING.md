# Contributing to TensorRT-Model-Connect

Thank you for your interest in contributing. External contributors should do all
development in a personal fork and submit changes to
[`NVIDIA/TensorRT-Model-Connect`](https://github.com/NVIDIA/TensorRT-Model-Connect)
through a pull request. Do not work directly on the upstream `main` branch.

The [Contributor Quickstart](website/docs/extend/contributing.md) and the
[Developer Guide](website/docs/developer-guide/overview.md) provide additional
project-specific design, testing, and ownership guidance.

## Development workflow

### 1. Fork and clone the repository

Use GitHub's **Fork** button to create
`https://github.com/YOUR-GITHUB-USERNAME/TensorRT-Model-Connect`, then clone
your fork and add the NVIDIA repository as `upstream`:

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git remote add upstream https://github.com/NVIDIA/TensorRT-Model-Connect.git
git fetch upstream
```

In this layout, `origin` is your writable fork and `upstream` is the canonical
repository. Keep these roles separate.

### 2. Create a focused branch from current upstream `main`

Create a short-lived topic branch for each contribution. Do not develop on your
fork's `main`, because it may be behind upstream:

```bash
git switch -c docs/improve-contributing upstream/main
```

Use a concise branch name that describes one purpose, such as
`fix/qwen-tokenizer` or `docs/model-validation`.

### 3. Find the narrowest owner

Before editing, identify the component that owns the behavior. Model-specific
semantics should normally remain in that model family's directory; shared
infrastructure should contain only genuinely model-independent contracts. Start
with the relevant guide:

- [Add a Model Family](website/docs/extend/add-model-family.md)
- [Add an Optimized Runtime Implementation](website/docs/extend/add-optimized-runtime.md)
- [Add a Runtime Strategy](website/docs/extend/add-runtime-strategy.md)
- [Add a Config Schema](website/docs/extend/add-config-schema.md)
- [Validate a Model Contribution](website/docs/extend/model-validation.md)

Open an issue before investing in a large, cross-cutting, or user-visible design
when the intended ownership or approach is not already clear.

### 4. Make a reviewable change and validate it locally

Keep each pull request focused on one problem. Add or update focused tests when
behavior changes, and update documentation when users or developers will observe
the change.

Install the lightweight commit-time quality hooks once in each clone:

```bash
python3 -m pip install --requirement requirements/community-ci.txt
pre-commit install --install-hooks
```

On Windows, use `py -3 -m pip` in place of `python3 -m pip`.

The hooks trim trailing whitespace, ensure one final newline, validate YAML,
check Ruff, and verify clang-format. Pre-commit manages the Ruff and
clang-format environments on Linux, macOS, and Windows. There is no pre-push
hook: builds and the broader source-only CPU suite run automatically on the
pull request after the branch is pushed.

Start with repository consistency checks:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
git diff --check
```

Then run the smallest meaningful tests for the code you changed. Model work also
needs the declared model, runtime, hardware, and comparison evidence. For
documentation changes, run:

```bash
python3 tools/check_doc_file_references.py --strict website/docs
npm --prefix website ci
npm --prefix website run build
```

Do not weaken a test, oracle, or acceptance threshold to make a change pass. If
you believe a test is incorrect, explain the evidence in the pull request and
ask a maintainer to review it.

### 5. Commit with a clear message and DCO sign-off

Repository history uses short, imperative, Conventional Commit-style subjects,
for example:

```text
docs(contributing): clarify external workflow
fix(qwen): preserve tokenizer contract
test(ci): cover fork pull requests
```

Sign off every commit as described in [Signing Your Work](#signing-your-work):

```bash
git commit --signoff -m "docs(contributing): clarify external workflow"
```

### 6. Sync your branch and push it to your fork

Before opening the pull request, incorporate the current upstream `main` and
rerun the affected checks:

```bash
git fetch upstream
git rebase upstream/main
git push --set-upstream origin docs/improve-contributing
```

If you rebase a branch that you already pushed, never use an unguarded force
push. Use `git push --force-with-lease`, and coordinate before rewriting a
branch that other people are using.

### 7. Open a pull request against upstream `main`

Open the pull request from your fork branch to
`NVIDIA/TensorRT-Model-Connect:main`. Include:

- the problem, scope, and important non-goals;
- a linked issue or design discussion when applicable;
- exact validation commands and their results;
- the model revision, artifacts, environment, and hardware for GPU claims; and
- tests or paths that were not run, plus any remaining risk.

Compilation, unit tests, inference, model parity, target-hardware execution,
performance, and release qualification are separate evidence levels. Claim only
what the recorded validation proves.

### 8. Run contributor-visible public CPU validation

Opening a pull request or pushing a new commit automatically starts Community
CPU against GitHub's exact pull-request merge revision. Separate jobs run
source quality, ownership and impact analysis, and the selected source-only C++
and Python units. No comment or maintainer action is required.

All public jobs run on GitHub-hosted `ubuntu-24.04` runners. Test jobs have
read-only repository permission and no access to private runners, secrets, or
GPUs. GitHub publishes native pull-request checks and public Actions logs,
including the complete output for every failed command.

Wait for `Community CPU / Required` to pass on the current merge revision. A
new commit automatically validates the new merge revision and cancels an older
in-progress run for the same pull request. If `main` advances and GitHub asks
for an update, rebase or update the branch so the new exact merge is validated.

### 9. Ask a maintainer to trigger protected CI

Opening a pull request or pushing to your fork does **not** start the protected
premerge suite. After public CPU validation passes and the pull request is
ready for protected CI, mention the repository maintainer in a pull-request
comment:

```text
@yifeif-nv This PR is ready for CI. Please trigger CI for the current head.
```

The maintainer verifies the pull-request head and a successful Community CPU
run for that head, then applies the one-shot `run-internal-ci` label. Only
collaborators with repository `maintain` or `admin` permission can authorize
that trigger.
The trusted bridge consumes the label, rechecks `Community CPU / Required`,
captures the current PR head SHA, and dispatches protected premerge validation.
If authorization rejects the request, ask the maintainer to remove and re-add
the retained label after satisfying the reported prerequisite. Adding an
already-present label does not create a new trigger event.

Wait for `trtmc/premerge/required` to pass on the exact pull-request head SHA.
If you push another commit, the previous result no longer validates the current
head; finish the update, wait for automatic Community CPU validation, and
mention `@yifeif-nv` once to request a new protected run. Private runner details,
logs, artifacts, and URLs are not part of the public contribution interface.

### 10. Respond to review and keep evidence current

Address review feedback on the same topic branch and sign off every new commit.
Keep the pull request current with upstream when requested. A new head requires
a fresh public merge result and a fresh maintainer-triggered protected result
before the pull request can merge. Maintainers merge accepted changes according
to the repository ruleset.

## Established development practices

The repository's development history consistently favors these practices:

- keep commits and pull requests single-purpose and easy to review;
- use an imperative Conventional Commit-style subject with a useful scope;
- keep model-owned behavior and its tests close to the model owner;
- pair behavior changes with focused regression coverage;
- record exact commands, revisions, artifacts, hardware, and untested paths;
- preserve copyright, license, attribution, and third-party provenance; and
- treat CI success as evidence for the exact tested revision, not for later
  commits or unexecuted configurations.

## Licensing and attribution

Contributions accepted into this project are licensed under the Apache License
2.0 unless explicitly stated otherwise.

Preserve all existing copyright, license, and attribution notices. If a
contribution incorporates or is derived from third-party material, identify its
source, version, and license in the pull request and include any notices required
for redistribution. Do not submit material unless you have the right to do so
under a license compatible with this project.

The DCO sign-off requirement is enforced prospectively for commits introduced
by pull requests and protected-branch updates after its adoption. Existing Git
history is not rewritten.

## Signing Your Work

- We require all contributors to sign off their commits. This certifies that
  the contribution is your original work, that you have the right to submit it
  under the same license, or that it uses a compatible license.
  - Contributions containing commits without a `Signed-off-by` line are not
    accepted.
- To sign off a commit, use the `--signoff` (or `-s`) option:

  ```bash
  git commit --signoff -m "docs(contributing): clarify external workflow"
  ```

  This appends a line like this to the commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```
- The full text of the [Developer Certificate of
  Origin](https://developercertificate.org/) follows:

  ```
    Developer Certificate of Origin
    Version 1.1
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.
    Developer's Certificate of Origin 1.1
    By making a contribution to this project, I certify that:
    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or
    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or
    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified it.
    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
