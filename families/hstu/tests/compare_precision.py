# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare held-out MovieLens ranking quality using matched native Task calls.

Training excludes the final interaction, which supplies the held-out target.
This sampled-candidate smoke evaluation is not production quality qualification.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from families.hstu.checkpoint import load_weights
from families.hstu.config import Config
from families.hstu.triton.native_client import NativeClient

GATES = {
    "max_relative_ndcg_loss": 0.01,
    "max_absolute_hit_rate_loss": 0.01,
    "min_mean_top10_overlap": 0.98,
    "max_mean_abs_score_error": 0.02,
}


def compare(root, runtime, ratings, examples=256, seed=20261124):
    rng = np.random.default_rng(seed)
    data = np.loadtxt(ratings, dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 4 or np.any(data[:, :2] < 1):
        raise ValueError("expected MovieLens user/item/rating/timestamp rows")
    users = {}
    for user, item, _, timestamp in data:
        users.setdefault(int(user), []).append((int(timestamp), int(item)))
    histories = [
        np.array([item for _, item in sorted(rows)]) for rows in users.values() if len(rows) >= 3
    ]
    selected = rng.choice(len(histories), examples, replace=False)
    result = []
    for count in (50, 64):
        c = Config.load(root / f"hstu{count}-bf16")
        fp_config = Config.load(root / f"hstu{count}-fp32")
        fp_weights = load_weights(root / f"hstu{count}-fp32", fp_config)
        bf_weights = load_weights(root / f"hstu{count}-bf16", c)
        for name in fp_weights:
            np.testing.assert_array_equal(fp_weights[name], bf_weights[name])
        rows = []
        with (
            NativeClient(root / f"hstu{count}-fp32/model.bundle", runtime) as fp32,
            NativeClient(root / f"hstu{count}-bf16/model.bundle", runtime) as bf16,
        ):
            for start in range(0, len(selected), 4):
                batch = selected[start : start + 4]
                inputs = {
                    "history_ids": np.zeros((len(batch), c.history_length), np.int64),
                    "history_lengths": np.zeros((len(batch), 1), np.int64),
                    "candidate_ids": np.zeros((len(batch), count), np.int64),
                    "num_candidates": np.full((len(batch), 1), count, np.int64),
                }
                for row, user in enumerate(batch):
                    sequence = histories[user]
                    context = sequence[max(0, len(sequence) - 1 - c.history_length) : -1]
                    target = int(sequence[-1])
                    negatives = rng.choice(
                        np.delete(np.arange(1, c.vocab_size), target - 1),
                        count - 1,
                        replace=False,
                    )
                    inputs["history_ids"][row, : len(context)] = context
                    inputs["history_lengths"][row, 0] = len(context)
                    inputs["candidate_ids"][row] = np.r_[target, negatives]
                reference, actual = fp32.score(inputs), bf16.score(inputs)
                errors = np.abs(actual["scores"] - reference["scores"])
                for row, (a, b) in enumerate(
                    zip(reference["picked_indices"], actual["picked_indices"])
                ):

                    def ndcg(picks):
                        rank = np.where(picks[:10] == 0)[0]
                        return float(1 / np.log2(rank[0] + 2)) if len(rank) else 0.0

                    rows.append(
                        dict(
                            fp32_ndcg=ndcg(a),
                            bf16_ndcg=ndcg(b),
                            top10_overlap=len(set(a[:10]) & set(b[:10])) / 10,
                            position_agreement=float(np.mean(a == b)),
                            top1_agreement=bool(a[0] == b[0]),
                            mean_abs_score_error=float(errors[row].mean()),
                            max_abs_score_error=float(errors[row].max()),
                        )
                    )

        def mean(name):
            return float(np.mean([r[name] for r in rows]))

        fp_ndcg, bf_ndcg = mean("fp32_ndcg"), mean("bf16_ndcg")
        fp_hit = float(np.mean([r["fp32_ndcg"] > 0 for r in rows]))
        bf_hit = float(np.mean([r["bf16_ndcg"] > 0 for r in rows]))
        summary = dict(
            candidates=count,
            examples=len(rows),
            seed=seed,
            fp32_ndcg_at_10=fp_ndcg,
            bf16_ndcg_at_10=bf_ndcg,
            relative_ndcg_loss=(fp_ndcg - bf_ndcg) / fp_ndcg if fp_ndcg else 0,
            fp32_hit_rate_at_10=fp_hit,
            bf16_hit_rate_at_10=bf_hit,
            mean_top10_overlap=mean("top10_overlap"),
            mean_greedy_position_agreement=mean("position_agreement"),
            top1_agreement=mean("top1_agreement"),
            mean_abs_score_error=mean("mean_abs_score_error"),
            max_abs_score_error=max(r["max_abs_score_error"] for r in rows),
            gates=GATES,
            rows=rows,
        )
        summary["passed"] = (
            summary["relative_ndcg_loss"] <= GATES["max_relative_ndcg_loss"]
            and fp_hit - bf_hit <= GATES["max_absolute_hit_rate_loss"]
            and summary["mean_top10_overlap"] >= GATES["min_mean_top10_overlap"]
            and summary["mean_abs_score_error"] <= GATES["max_mean_abs_score_error"]
        )
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared output directory")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    result = compare(args.root, args.runtime_root, args.ratings, args.examples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in result], indent=2))
    return 0 if all(r["passed"] for r in result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
