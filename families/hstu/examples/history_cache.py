# SPDX-License-Identifier: Apache-2.0
"""Export, destroy the producer, restore, and score without history computation."""

import argparse
from pathlib import Path
import numpy as np
from families.hstu.config import Config
from families.hstu.reference import example_inputs
from families.hstu.triton.native_client import NativeClient, save_snapshot, snapshot_metadata


def load_inputs(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: np.array(values[k], copy=True) for k in values.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    a = p.parse_args()
    x = example_inputs(Config.load(a.checkpoint), 1)
    with NativeClient(a.bundle, a.runtime_root) as producer:
        snapshot = producer.export_history(
            {k: v for k, v in x.items() if k.startswith("history_")}, ["example-v1"]
        )
        cold = producer.score(x)
    save_snapshot(a.snapshot, snapshot)
    with NativeClient(
        a.bundle, a.runtime_root, hydration=True, kv_cache_size_bytes=8 << 20
    ) as consumer:
        handles = consumer.restore_history(a.snapshot.read_bytes())
        result = consumer.score_with_history(
            handles,
            snapshot_metadata(snapshot)["revisions"],
            {k: v for k, v in x.items() if not k.startswith("history_")},
        )
        for k in cold:
            np.testing.assert_array_equal(result[k], cold[k])
        assert consumer.stats()["history_prepare_calls_total"] == 0
        print("Restored scoring matches cold scoring exactly; no history preparation.")
        print(consumer.stats())
        consumer.release_history(handles)


if __name__ == "__main__":
    main()
