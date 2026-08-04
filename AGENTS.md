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

<!-- Collaborative review anchor. -->
