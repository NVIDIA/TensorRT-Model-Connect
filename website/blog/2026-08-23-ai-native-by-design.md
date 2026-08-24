---
slug: ai-native-by-design
title: "AI-Native by Design: What We Learned Building TensorRT-Model-Connect"
description: "How parallel work, model-family isolation, reversible changes, and GPU-backed validation shaped a project designed around coding agents."
image: /img/blog/ai-native-by-design/ai-native-by-design-hero.png
authors:
  - yifei
tags:
  - AI-native development
  - coding agents
  - software architecture
  - validation
---

import Diagram from '@site/src/components/Diagram';

*How parallel work, model-family isolation, reversible changes, and GPU-backed
validation shaped an open-source project designed around coding agents.*

![Editorial illustration of a human directing many blue candidate paths through red and amber validation gates toward one green trusted result](/img/blog/ai-native-by-design/ai-native-by-design-hero.png)

*Many untrusted candidates enter the system. Architecture, independent
challenge, and evidence determine what deserves to emerge.*

[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
began with a practical question: could we make the performance of NVIDIA's
inference stack accessible to model developers who are not TensorRT experts?

I initially approached the project as an experiment with coding agents. Within
the first few days, however, I became more interested in a larger question:
what would it mean to design a serious software project around AI agents from
the beginning—not merely use an agent to accelerate an existing development
process?

Our answer has not been an elaborate orchestration system or an ever-growing
collection of prompts. It has been a set of engineering choices:

- choose work that can scale horizontally;
- give agents outcomes and objective references instead of prescribing every
  implementation step;
- isolate model-family changes so failures remain local;
- make changes easy to evaluate and revert; and
- treat automated validation as the production constraint.

That is the sense in which Model Connect is AI-native. AI increases the rate at
which we can produce candidate implementations. Architecture and validation
determine whether that increased output becomes reliable software.

<!-- truncate -->

## What “AI-native” means in this project

“AI-native” can mean many things. Here I use it in a narrow, operational sense:

> An AI-native project is structured so meaningful engineering tasks can be
> performed independently, evaluated against explicit evidence, and safely
> accepted or rejected without destabilizing the wider system.

This does not mean that AI writes everything. It does not mean that human
judgment disappears. And it does not mean that every software project should
adopt the same model.

By “software factory,” I do not mean code emitted without supervision. I mean
a production system capable of exploring many candidate changes and subjecting
each one to repeatable quality control. Compute helps create the candidates.
Tests, reference comparisons, benchmarks, and human review determine what is
ready to ship.

## Start with work that can scale horizontally

Some engineering workloads have a long serial critical path. Others expose
many independent workstreams. Adding agents helps far more in the second
category.

The long tail of AI models is a natural fit for horizontal work. Model
families, configurations, operators, runtime paths, and validation cases can
often be investigated independently. Work on one model family does not always
need to block work on another.

That problem geometry is central to Model Connect. The project provides
family-owned reference implementations that turn supported Hugging Face or
local checkpoints into versioned `.bundle` artifacts, then expose task-oriented
native C++ APIs for text, vision, audio, diffusion, segmentation, embedding,
forecasting, and other workloads. The build and runtime boundary is documented
in the
[Project Overview](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/project-overview).

As of the public July 29, 2026 release comparison, the project covered 105
unique single-process release profiles across 76 model families on NVIDIA
GB300. That number is not, by itself, a measure of agent productivity. It does
show why the problem benefits from an architecture that can grow by adding
independent units rather than extending one serial integration path.

The first lesson is therefore simple:

> AI-native development begins with problem selection. If work cannot be
> separated, adding more agents mostly adds coordination.

## Give agents outcomes and references, not recipes

Most Model Connect agent runs begin with an outcome: support a model family,
close an accuracy gap, improve a performance path, or strengthen a contract.
We also identify the evidence required to accept the result—often behavior
from an established reference implementation, plus project-specific tests and
constraints.

We intentionally began with a simple outer loop: a high-level goal, a
general-purpose coding agent, repository instructions, and strict validation.
Today, people still initiate most long-running tasks. We generally avoid
prescribing the full implementation plan unless the task or an observed
failure mode requires it.

Our working hypothesis is that a capable general-purpose agent benefits from
room to use patterns it has already learned. Instead of encoding an engineer's
preferred implementation into every prompt, we specify the result, the
boundaries, and the evidence required.

The implementation path is flexible. The acceptance criteria are not.

This is minimal orchestration, not minimal control. The agent may explore,
implement, test, fail, and revise inside an isolated task. The resulting change
must still satisfy the same architectural and technical gates as any other
contribution. We add constraints when repeated evidence shows they are needed,
not simply because a workflow can be hard-coded.

## Make isolation the unit of scale

The most important constraint on AI-native development has not been the
agent's ability to complete an individual task. It has been the architecture
surrounding that task.

For Model Connect, we separated components that evolve at different speeds:

- **TensorRT and CUDA form the stable execution foundation.** Compatibility,
  performance, reliability, and long-term contracts matter here.
- **Model Connect is the faster-moving integration layer.** It connects a
  broad and rapidly changing model ecosystem to that foundation.
- **Model-family implementations own model-specific knowledge.** Builders,
  runtime pipelines, helper kernels, configuration, and validation evidence
  stay with the family that needs them.

This design deliberately favors independence. Similar model families may
contain some duplication because independence is itself a scaling feature. A
shared abstraction can reduce lines of code, but it can also couple unrelated
work, increase merge conflicts, and enlarge the blast radius of a mistake.

We therefore promote behavior into shared infrastructure only when multiple
independent owners need the same assumption-free contract. Everything else
stays close to the model family that owns it. The public
[Units and Ownership](https://nvidia.github.io/TensorRT-Model-Connect/architecture/units-and-ownership)
documentation makes those boundaries explicit.

<Diagram
  src="/img/blog/ai-native-by-design/isolation-architecture.svg"
  alt="Three independent model-family implementations connected through the Model Connect interface to a stable TensorRT and CUDA foundation, with one local failure contained inside its family"
  caption="A narrow shared contract lets model-family work scale without requiring every implementation to move together."
/>

Isolation does not eliminate all systemic risk: shared build, runtime,
packaging, and CI infrastructure can still affect multiple families. But it
materially reduces the number of changes that must move together and makes
parallel work safer.

We pair isolation with reversibility. We prefer two-way doors: changes that are
easy to evaluate, easy to revert, and unlikely to cascade into unrelated model
families. That lets us learn quickly without confusing speed with permission to
weaken the system.

## When candidate code becomes cheaper, evidence becomes more expensive

AI makes candidate implementations cheap. It does not make correctness cheap.
The scarce output is evidence that can survive both automation and human
judgment.

**Make evidence human-legible.** Machine checks are necessary, but a reviewer
cannot quickly interpret a pile of tensors or raw values. Semantic task
interfaces—text in/text out or text in/image out—make final behavior legible
enough for a human to spot-check quickly. A spot check is not proof; it
complements automated tests by ensuring that their evidence ends in behavior a
person can understand.

**Make validation agent-native and self-improving.** Agents can generate tests,
probes, and operating procedures alongside the code, then refine them as real
artifacts expose missing assumptions. If automated checks pass but a human
finds a bad final artifact, the process admitted a false success. We reproduce
the failure, encode the missing invariant or regression, and harden the SOP so
the next run is harder to fool.

**Make QA and development adversarial collaborators.** QA is not a downstream
team that receives a finished implementation. QA and developers operate on the
same reproducible CI pipeline from organizationally independent positions: QA
acts as a red team that tries to falsify the implementation's claims;
developers harden the implementation and the pipeline in response. Shared
evidence makes findings reproducible. Independent ownership keeps the challenge
credible.

> Candidate code can scale with agents and tokens. Trustworthy software can
> scale only as fast as its evidence and validation system.

<Diagram
  src="/img/blog/ai-native-by-design/software-factory.svg"
  alt="Flow from human intent through a high-level goal, parallel agent runs, validation, and a verified change, with failed candidates returning for refinement or reversion"
  caption="Humans own intent and release. Agents explore. Evidence decides."
/>

## Human judgment moves up a level

The practical effect is that every engineer takes on work that resembles
management and direction. The highest-leverage questions move upstream:

- What problem is worth solving?
- Can the work be decomposed and scaled safely?
- What technical and organizational constraints can turn untrusted candidate
  outputs into a result that deserves trust?

Agent outputs begin as untrusted candidates. Model-family ownership, reversible
changes, independent QA challenge, reproducible CI, and human-legible evidence
do not guarantee correctness. They make claims falsifiable, failures easier to
contain, and acceptance or rejection easier to review.

Humans still inspect implementations and debug failures. But their most
valuable work increasingly lies in designing and governing the system: setting
intent and acceptance criteria, deciding where independence is required,
interpreting anomalous evidence, and remaining accountable for release.

## What we have not solved

Model Connect is a public preview, and several parts of this development model
remain working hypotheses.

- Not every engineering task can be decomposed into independent units.
- Minimal project-specific orchestration is not a universal best practice; we
  expect to add structure where repeated failures justify it.
- Model-family isolation reduces blast radius but cannot eliminate failures in
  shared infrastructure.
- Reference implementations are useful comparison points, not infallible
  oracles. Tests also need independent invariants and carefully reviewed
  tolerances.
- More parallel agents can increase demand for validation faster than they
  increase accepted throughput.
- Most tasks are still initiated by people. Automated task discovery and
  large-scale concurrency are future directions, not claims about the current
  system.

These limitations are not incidental. They define the engineering work
required to make AI-native development dependable.

## From a production model to a better developer experience

The purpose of this work is not the production system itself. It is the
developer experience that the system can make possible.

Model developers should not need to become inference experts before they can
evaluate and deploy a supported model efficiently on NVIDIA hardware. Model
Connect aims to provide a clear path from a Hugging Face or local checkpoint
to a versioned bundle and native task API, while keeping the model-family
implementation visible enough to inspect, extend, and customize.

Our longer-term aspiration is straightforward: connect a model through a
stable boundary, then continue benefiting as TensorRT, CUDA, kernels,
compilers, and supported NVIDIA platforms improve underneath it. This is an
aspiration, not a promise that every model or target is supported today. Exact
support and qualification evidence remain the source of truth.

TensorRT-Model-Connect will succeed only if it lowers the expertise barrier
while preserving the accuracy, performance, reliability, and maintainability
developers expect.

That is also the larger promise of AI-native development: not code for its own
sake, but a way to make previously expensive, fragmented engineering problems
economically possible—without giving up evidence, accountability, or quality.

## Try it and help us improve it

TensorRT-Model-Connect is open source and evolving rapidly. You can:

- follow the
  [Quick Start](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/quick-start)
  to build and run a supported model;
- explore the
  [Supported Models](https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/overview)
  and their qualification evidence;
- read the
  [AI and Agent Guide](https://nvidia.github.io/TensorRT-Model-Connect/agent-guide);
  or
- open an issue or contribute through the
  [GitHub repository](https://github.com/NVIDIA/TensorRT-Model-Connect).

We are still learning what an AI-native open-source project should look like.
The most valuable feedback will come from developers who try the system,
inspect its evidence, find its limits, and help us improve the boundaries.
