# Agent Instructions

## Repository Target

- Treat GitHub as the active repository for this project:
  `https://github.com/NVIDIA/TensorRT-Model-Connect.git`.
- Use the local `github` remote for fetch, push, PR, and CI operations.
- Do not push project changes to any non-GitHub legacy remote unless the user
  explicitly asks for that repository.

## Branch And PR Flow

- The GitHub default branch is `main`.
- Do not push directly to GitHub `main`.
- Start new work from `github/main` on a short-lived branch.
- Push the branch to the GitHub remote and open a pull request targeting
  `main`.
- Wait for GitHub CI before merging.
- Merge with squash or rebase, matching the repository ruleset.
- Avoid commit messages containing `Claude`; the GitHub ruleset rejects them.

## GitHub Pages

- Keep GitHub Pages dedicated to the documentation website.
- Do not publish CI reports to GitHub Pages unless the user explicitly changes
  that decision.

## Model-Family Ownership And Execution Philosophy

- The model family is the unit of ownership, fault isolation, and horizontal
  scale. One team or agent must be able to implement, validate, change, and
  revert one family without changing or coordinating with another.
- Each model family owns all model-specific code: configuration and weights,
  mathematical topology, engine composition, runtime orchestration, and
  model validation.
- Duplicate model-specific code by design. Duplication keeps model-specific
  defects, merge conflicts, and rollbacks family-local; code similarity never
  justifies a cross-family abstraction.
- Shared code is limited to model-agnostic contracts and mechanics. It contains
  no model topology, model semantics, model orchestration, model-specific
  validation logic or evidence, or family-specific behavior.
- Generic family code defines only the mathematical computation graph and
  runtime orchestration. TensorRT owns all lowering to GPU execution, including
  fusion, tactic and kernel selection, scheduling, code generation, and
  hardware or version adaptation.
- Generic family semantics do not depend on GPU, SM, CUDA, driver, or TensorRT
  version. Platform-specific failures are project topology or orchestration
  defects, or upstream TensorRT issues.
- Platform specialization is limited to complete network offload, such as
  TensorRT Edge-LLM, and TVM-FFI BYOK kernel bindings. Target-specific plans,
  timing caches, and compiled kernels do not make the model family specialized.
- Existing family-owned GPU helper kernels outside TVM-FFI BYOK violate this
  architecture and are temporary migration debt, not a supported specialization.

## Dos And Don'ts

- Do keep validation criteria meaningful and aligned with the behavior under
  test.
- Never change the test passing criteria for the purpose of passing CI. If you believe the test is faulty, escalate to a human

## Repo Skills

- Codex skills packaged for this repo are registered through
  `.agents/plugins/marketplace.json`.
- The `trtmc-agent-skills` plugin is marked `INSTALLED_BY_DEFAULT` and exposes
  repo-local skills from `plugins/trtmc-agent-skills/skills/`.
- Use `$write-git-messages` when drafting or reviewing commit messages, PR
  titles, PR descriptions, squash merge messages, or rebase message text.
- If `$write-git-messages` is not listed in the active runtime skills, load
  `plugins/trtmc-agent-skills/skills/write-git-messages/SKILL.md` directly and
  follow it.

<!-- Collaborative review anchor: batch 2. -->
