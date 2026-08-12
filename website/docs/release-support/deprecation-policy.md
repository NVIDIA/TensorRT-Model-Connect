---
title: Deprecation Policy
description: How stable user-facing interfaces will be announced, migrated, and removed after releases begin.
---

The project has not yet established a released compatibility window. This page
therefore defines the documentation shape to use once stable releases begin;
it does not promise a window retroactively.

A future deprecation should include:

- the affected CLI flag, API, config field, bundle contract, or manifest field;
- the first release that warns;
- the supported replacement and migration example;
- the earliest release in which removal can occur; and
- tests that keep the warning and replacement behavior meaningful.

Do not weaken validation criteria to preserve a deprecated path. If a test no
longer represents the intended contract, change the contract and test through
normal human review with a migration note.

{/* Collaborative review anchor: batch 2. */}
