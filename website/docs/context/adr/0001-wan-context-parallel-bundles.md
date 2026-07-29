---
number: 0001
title: Add padded-head Ulysses context-parallel Wan bundles
status: Proposed
date: 2026-07-29
source_commits: []
---

## Context

Wan2.1 text-to-video inference spends most denoiser work over the patch-token
sequence. Tensor parallelism shards weights and heads, but it does not provide a
sequence-sharding mode and requires one rank-specific engine per GPU. Wan2.1
1.3B has 12 attention heads, so equal Ulysses head sharding cannot directly
support the requested CP8 configuration.

The bundle schema also needs to distinguish context parallelism from existing
tensor-parallel bundles without allowing decoder builders to accept the new
mode accidentally.

## Decision

Add `context_parallel` to the distributed build metadata and expose CP sizes
2, 4, and 8 through the build CLI. Tensor and context parallelism remain
mutually exclusive. Context-parallel diffusion bundles store one shared
rank-dynamic denoiser plan in `denoiser_plan_cp` and persist the selected CP
world size in bundle configuration.

Wan keeps weights, timestep inputs, and text context replicated. The denoiser
uses REDUCE_SCATTER to select rank-local patch and rotary rows. Self-attention
uses Ulysses ALL_TO_ALL exchanges to convert local sequence shards into
full-sequence head shards and back. CP2 and CP4 route the 12 model heads
directly. CP8 appends four zero-only heads, routes 16 heads evenly, and removes
the padded heads after attention. Cross-attention queries remain local against
replicated text context. A final ALL_GATHER restores all patch rows and
preserves the existing runtime ABI. The existing `diffusion_wan` runtime
strategy initializes the distributed communicator and only rank zero decodes
and writes video artifacts.

## Considered Alternatives

- Rejecting CP8 because 12 heads are not divisible by eight was rejected;
  padding to 16 routed heads preserves the 12 real attention outputs while
  supporting the requested world size.
- Gathering full K/V rows while keeping local queries was rejected because it
  duplicated attention heads and communicated more activation data than
  Ulysses.
- Building one plan per CP rank was rejected because the collective layers can
  determine rank dynamically and replicated weights make the graphs identical.
- Reusing tensor-parallel metadata for context parallelism was rejected because
  it would misrepresent the execution contract and could route unsupported
  decoder builders into a sequence-parallel path.

## Consequences

- Wan2.1 1.3B supports CP2, CP4, and CP8 with one serialized denoiser plan.
- The number of patch tokens must be divisible by the CP world size.
- TensorRT 11 or newer and an MPI-launched distributed runtime are required.
- Weights remain replicated, so CP reduces activation and query-side compute
  rather than model-weight memory.
- CP8 computes four zero-only attention heads before discarding them, trading
  modest redundant work for even head routing.
- Single-device and tensor-parallel bundle layouts remain unchanged.
