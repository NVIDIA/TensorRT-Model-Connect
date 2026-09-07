---
title: Traceability and Safety Status
---

:::warning No functional-safety certification claim

This repository does not claim ISO 26262 certification, ISO 26262-6
compliance, or complete bidirectional requirements-to-test traceability. This
page describes evidence boundaries; it is not a safety case, legal advice, or
functional-safety approval.

:::

## Current traceability sources

The repository provides practical ownership and test traceability through:

- one physical `families/<family>/` directory per owner;
- exact checkpoint and task declarations in `support.py` and family manifests;
- family-owned Python, C++, and E2E tests;
- shared-core tests for bundle bounds, discovery, loading, and public APIs; and
- change-impact selection based on the physical family path.

These sources make a claim inspectable, but they are not a formal requirements
database or a complete safety traceability matrix. Counts and support state
change with the tree, so the generated model inventory and exact test results
must be used instead of a copied snapshot.

## Evidence levels

| Evidence | What it proves | What it does not prove |
| --- | --- | --- |
| Source and ownership checks | The tree follows the declared dependency boundaries | GPU execution or model accuracy |
| Unit tests | The tested contract behaves as asserted | Whole-model support |
| Family E2E | One exact manifest and testcase pass | Every checkpoint, option, or target |
| Target qualification | The tested hardware/software tuple works | Other targets or revisions |
| Performance result | A declared workload and timing boundary were measured | General performance or functional safety |

## Requirements for a stronger claim

A formal traceability or safety claim would additionally require maintained
requirement identifiers, links in both directions, uniqueness and coverage
checks, retained exact-revision evidence, controlled review and anomaly
management, tool qualification where applicable, and an approved safety scope
and assessment process.

Until those controls exist, describe repository-wide traceability as partial
and support claims as exact to their family manifest and evidence.
