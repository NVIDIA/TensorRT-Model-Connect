# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2 conversion-parity inputs: seeded single-chain proteins of 24 to 117 residues (the bundle's token profile)
from the pinned UniRef50 test shard (zgcarvalho/uniref50-test, CC-BY-4.0), each a single-sequence request (no MSA,
as the family's README evaluates) prepared for the bundle's static profile with the family's seeded noise. A
sequence the profile cannot hold (more padded atoms) is skipped. Run in the reference environment after
reference/prepare.py: ``python inputs.py --count N --output DIR`` prints the records as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from families.boltz2.tests.aiperf_requests import prepare

DATASET = ("zgcarvalho/uniref50-test", "0e0e4f88b99cf30f5ec2432c1c38c9908a08e5e8",
           "data/test-00000-of-00008-c0f2d5a0ed5fe5d1.parquet")
SEED = 20261113
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repo, revision, filename = DATASET
    table = pq.read_table(hf_hub_download(repo, filename, repo_type="dataset", revision=revision), columns=["sequence"])
    sequences = sorted({sequence for sequence in table.column("sequence").to_pylist()
                        if 24 <= len(sequence) <= 117 and set(sequence) <= AMINO_ACIDS})
    random.Random(SEED).shuffle(sequences)
    records, skipped = [], []
    for index, sequence in enumerate(sequences):
        if len(records) == arguments.count:
            break
        request = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n"
        try:
            document, _ = prepare(request.encode(), arguments.output / f"protein-{index:05d}")
        except ValueError as error:  # outside the static profile (more padded atoms than it holds)
            skipped.append(f"{len(sequence)} residues: {error}")
            continue
        records.append({"id": f"uniref50-test/{index}", "cluster": f"uniref50-test/{index}",
                        "request": {"document_path": str(document)}, "residues": len(sequence)})
    if len(records) < arguments.count:
        raise RuntimeError(f"only {len(records)} of {arguments.count} sequences fit the Boltz-2 profile; "
                           f"last refusal: {skipped[-1] if skipped else None}")
    print(json.dumps(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
