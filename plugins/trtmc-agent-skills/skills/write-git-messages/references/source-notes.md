# Source Notes

Use these notes when a user asks why the skill recommends a format or when local repository convention conflicts with a default.

## Primary Sources Researched

- Git user manual, "Creating good commit messages": recommends a short first line, blank line, then a fuller description; Git treats text before the first blank line as the commit title.
  - https://git-scm.com/docs/user-manual
- Conventional Commits 1.0.0: defines `<type>[optional scope]: <description>`, optional body, and optional footer structure; uses `fix`, `feat`, and `BREAKING CHANGE` to communicate semantic intent, while allowing other types.
  - https://www.conventionalcommits.org/en/v1.0.0/
- GitHub Docs, "Helping others review your changes": recommends small focused PRs, clear titles/descriptions, purpose, change overview, links to context, requested feedback, review order for multi-file PRs, and self-review/build/test before submitting.
  - https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes

## Practical Interpretation

- Commit messages optimize long-term history search and release automation.
- PR messages optimize review speed, reviewer confidence, and collaboration.
- A good PR title often becomes the squash merge title, so it should be written with the same care as a commit summary.
- Local repository convention wins over generic style unless it is unclear, missing, or explicitly being replaced.
