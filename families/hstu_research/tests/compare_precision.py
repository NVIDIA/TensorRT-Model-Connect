# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relative precision on sampled MovieLens-20M candidates, not publisher metric reproduction."""

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from families.hstu_research.checkpoint import digest
from families.hstu_research.config import Config
from families.hstu_research.triton.native_client import NativeClient


def data(archive, limit):
    with zipfile.ZipFile(archive) as z:
        catalog = [
            int(r["movieId"]) for r in csv.DictReader(io.TextIOWrapper(z.open("ml-20m/movies.csv")))
        ]
        users = {}
        for row in csv.DictReader(io.TextIOWrapper(z.open("ml-20m/ratings.csv"))):
            user = int(row["userId"])
            if user > limit:
                break
            users.setdefault(user, []).append((int(row["timestamp"]), int(row["movieId"])))
    return catalog, [sorted(v) for v in users.values() if len(v) >= 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--ratings-archive", type=Path, required=True)
    parser.add_argument("--users", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert digest(args.fp32 / "hstu_research_weights.npz") == digest(
        args.bf16 / "hstu_research_weights.npz"
    )
    c, low = Config.load(args.fp32), Config.load(args.bf16)
    assert c.architecture == low.architecture and c.serving == low.serving
    catalog, users = data(args.ratings_archive, args.users)
    assert users and max(catalog) < c.architecture.vocab_size
    names = ["fp32", "bf16"]
    clients = [NativeClient(p / "model.bundle", args.runtime_root) for p in (args.fp32, args.bf16)]
    assert [task.metadata["precision"] for task in clients] == names
    batch = clients[0].metadata["max_batch_size"]
    if c.serving.batching == "fixed" and len(users) % batch:
        raise ValueError("selected user count must fill the fixed batch")
    t, count = c.serving.history_length, c.serving.candidates
    rng = np.random.default_rng(20261124)
    catalog = np.array(catalog, np.int64)
    rows = []
    for start in range(0, len(users), batch):
        group = users[start : start + batch]
        b = len(group)
        x = dict(
            history_ids=np.zeros((b, t), np.int64),
            history_lengths=np.zeros((b, 1), np.int64),
            history_timestamps=np.zeros((b, t + 1), np.int64),
            candidate_ids=np.zeros((b, count), np.int64),
            num_candidates=np.full((b, 1), count, np.int64),
        )
        for row, sequence in enumerate(group):
            prefix = sequence[-t - 1 : -1]
            target = sequence[-1]
            ids = np.array([i for _, i in prefix], np.int64)
            n = len(prefix)
            x["history_ids"][row, :n] = ids
            x["history_lengths"][row, 0] = n
            x["history_timestamps"][row, :n] = [ts for ts, _ in prefix]
            x["history_timestamps"][row, n] = target[0]
            pool = catalog[~np.isin(catalog, np.r_[ids, target[1]])]
            x["candidate_ids"][row] = np.r_[target[1], rng.choice(pool, count - 1, replace=False)]
        outputs = [task.score(x) for task in clients]
        reference = outputs[0]["scores"]
        ranks = [np.argsort(-out["scores"], axis=1, kind="stable") for out in outputs]
        for row in range(b):
            rank = [int(np.flatnonzero(r[row] == 0)[0]) for r in ranks]
            rows.append(
                {
                    "rank": rank,
                    "top10_overlap": [
                        len(set(ranks[0][row, :10]) & set(r[row, :10])) / 10 for r in ranks[1:]
                    ],
                    "mae": [
                        float(np.mean(abs(out["scores"][row] - reference[row])))
                        for out in outputs[1:]
                    ],
                    "max_error": [
                        float(np.max(abs(out["scores"][row] - reference[row])))
                        for out in outputs[1:]
                    ],
                    "pointwise_pass": [
                        bool(
                            np.all(
                                abs(out["scores"][row] - reference[row])
                                <= 0.02 + 0.02 * abs(reference[row])
                            )
                        )
                        for out in outputs[1:]
                    ],
                }
            )
    summary = []
    for index, name in enumerate(names):
        ranks = np.array([r["rank"][index] for r in rows])
        ndcg = float(np.mean(np.where(ranks < 10, 1 / np.log2(ranks + 2), 0)))
        hit = float(np.mean(ranks < 10))
        d = dict(name=name, examples=len(rows), ndcg_at_10=ndcg, hit_rate_at_10=hit)
        if index:
            d.update(
                mean_top10_overlap=float(np.mean([r["top10_overlap"][index - 1] for r in rows])),
                mean_abs_error=float(np.mean([r["mae"][index - 1] for r in rows])),
                max_abs_error=max(r["max_error"][index - 1] for r in rows),
                all_pointwise_pass=all(r["pointwise_pass"][index - 1] for r in rows),
            )
        summary.append(d)

    result = dict(
        dataset="MovieLens 20M",
        dataset_sha256=digest(args.ratings_archive),
        sampling="first users, final interaction held out, C-1 sampled unseen catalog negatives",
        summary=summary,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for task in clients:
        task.close()
    print(json.dumps(summary, indent=2))
    assert all(r["pointwise_pass"][0] for r in rows)
    assert np.mean([r["top10_overlap"][0] for r in rows]) >= 0.9


if __name__ == "__main__":
    main()
