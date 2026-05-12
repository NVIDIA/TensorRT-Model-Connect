# Skills

Project-specific agent skills. Each skill is a `SKILL.md` file in its own
subdirectory.

Codex-discoverable skills are packaged in
`plugins/trtmc-agent-skills/skills/` and registered by
`.agents/plugins/marketplace.json`.

| Skill | Purpose |
|-------|---------|
| [profile-model](profile-model/SKILL.md) | E2E performance profiling — TRT vs HF vs torch.compile, per-layer timing, CPU phase breakdown, bottleneck classification |
| [debug-trt-mismatch](debug-trt-mismatch/SKILL.md) | 5-level numerical divergence investigation — logits, layers, VL, C++ parity, graph op isolation |
| [doc-sync](doc-sync/SKILL.md) | Daily documentation maintenance — ADR upkeep, wiki drift repair, traceability audit |
| [submit-gitlab-mr](submit-gitlab-mr/SKILL.md) | Push branch and create GitLab MR via glab, with automatic ADR creation |
| [mr-babysitter](mr-babysitter/SKILL.md) | Monitor GitLab MR CI pipelines, diagnose and fix failures |
| [gitlab-ai-staging-autopilot](gitlab-ai-staging-autopilot/SKILL.md) | Autonomous ai-staging MR queue — rebase green AI MRs, merge clean rebases, mark conflicts for rework |
| [ai-task-discovery](ai-task-discovery/SKILL.md) | Discover atomic, verifiable AI reliability tasks and publish them as GitLab issues |
| [ai-task-implementer](ai-task-implementer/SKILL.md) | Claim one ready AI task, implement it narrowly, and open an MR targeting ai-staging |
| [ai-staging-babysitter](ai-staging-babysitter/SKILL.md) | Snapshot ai-staging, reset it to master, and open the promotion MR |
| [ai-promotion-babysitter](ai-promotion-babysitter/SKILL.md) | Keep timestamped promotion MRs green for human review |
| [fp16-trt-network](fp16-trt-network/SKILL.md) | Guide for building FP16 TensorRT networks in strongly-typed mode |
| [optimize-model-precision](optimize-model-precision/SKILL.md) | Autonomous precision optimization — find best low-precision config |
