# Agent Instructions

## Repository Target

- Treat GitHub as the active repository for this project:
  `https://github.com/NVIDIA/TensorRT-Model-Connect.git`.
- Use the local `github` remote for fetch, push, PR, and CI operations.
- Do not push project changes to the GitLab `origin` remote unless the user
  explicitly asks for GitLab work.

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
