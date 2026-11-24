# SPDX-License-Identifier: Apache-2.0
"""Train the public one-block HSTU ranking adapter on a local MovieLens u.data.

This is an example checkpoint recipe, not an official Meta pretrained model.
Download/accept the dataset terms separately; data are never included in a bundle.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .checkpoint import create_fixture, load_weights
from .config import Config, REVISION, UPSTREAM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratings", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261124)
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 1:
        p.error("steps and batch size must be positive")
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = np.loadtxt(a.ratings, dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 4 or np.any(data[:, :2] < 1):
        raise ValueError("expected MovieLens user/item/rating/timestamp rows")
    users = {}
    for user, item, rating, timestamp in data:
        users.setdefault(int(user), []).append((int(timestamp), int(item)))
    histories = [
        np.array([item for _, item in sorted(rows)], np.int64)
        for rows in users.values()
        if len(rows) >= 3
    ]
    if not histories:
        raise ValueError("need users with at least three ordered interactions")
    create_fixture(a.output, seed=a.seed)
    c = replace(Config.load(a.output), vocab_size=int(data[:, 1].max()) + 1)
    # Regenerate only the publicly initialized item table for this dataset vocabulary.
    base = load_weights(a.output, Config.load(a.output))
    base["item_embeddings"] = rng.normal(0, 0.02, (c.vocab_size, c.hidden_size)).astype(np.float32)
    weights = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(torch.from_numpy(v)) for k, v in base.items()}
    )
    optimizer = torch.optim.AdamW(weights.parameters(), lr=1e-3, weight_decay=1e-4)
    c.save(a.output)
    d, h, t, r, k = c.hidden_size, c.num_heads, c.history_length, c.decode_steps, c.candidates

    def sample(batch, evaluate=False):
        ids = np.zeros((batch, t), np.int64)
        lengths = np.zeros(batch, np.int64)
        candidates = np.zeros((batch, k), np.int64)
        for row in range(batch):
            seq = histories[int(rng.integers(len(histories)))]
            index = len(seq) - 1 if evaluate else int(rng.integers(1, len(seq) - 1))
            context = seq[max(0, index - t) : index]
            ids[row, : len(context)] = context
            lengths[row] = len(context)
            target = int(seq[index])
            negative = rng.choice(
                np.delete(np.arange(1, c.vocab_size), target - 1), k - 1, replace=False
            )
            candidates[row] = np.concatenate([[target], negative])
        return torch.from_numpy(ids), torch.from_numpy(lengths), torch.from_numpy(candidates)

    def forward(ids, lengths, candidates):
        b = ids.shape[0]
        w = weights
        raw = torch.cat(
            [w["item_embeddings"][ids], torch.zeros((b, r, d)), w["item_embeddings"][candidates]],
            dim=1,
        )
        positions = torch.cat(
            [
                torch.arange(t).expand(b, -1),
                lengths[:, None] + torch.arange(r)[None, :],
                lengths[:, None].expand(-1, k),
            ],
            dim=1,
        )
        x = raw * 16 + w["position_embeddings"][positions]
        u, v, q, key = F.silu(F.layer_norm(x, (d,), eps=c.epsilon) @ w["uvqk"]).chunk(4, dim=-1)

        def heads(value):
            return value.reshape(b, -1, h, d // h).transpose(1, 2)

        query = heads(q[:, -k:])
        key = heads(key)
        v = heads(v)
        history = (torch.arange(t)[None, :] < lengths[:, None])[:, None, None, :].expand(b, 1, k, t)
        mask = torch.cat(
            [
                history,
                torch.zeros((b, 1, k, r), dtype=torch.bool),
                torch.eye(k, dtype=torch.bool)[None, None, :, :].expand(b, -1, -1, -1),
            ],
            dim=-1,
        )
        attn = F.silu(query @ key.transpose(-1, -2)) / (t + r + k) * mask
        attended = (attn @ v).transpose(1, 2).reshape(b, k, d)
        encoded = (
            (u[:, -k:] * F.layer_norm(attended, (d,), eps=c.epsilon)) @ w["output_weight"]
            + w["output_bias"]
            + x[:, -k:]
        )
        return (
            F.layer_norm(encoded, (d,), eps=c.epsilon) @ w["head_weight"] + w["head_bias"]
        ).squeeze(-1)

    losses = []
    for step in range(a.steps):
        logits = forward(*sample(a.batch_size))
        loss = F.cross_entropy(logits, torch.zeros(a.batch_size, dtype=torch.int64))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weights.parameters(), 1)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 20 == 0:
            print("step", step, "loss", losses[-1], flush=True)
    with torch.no_grad():
        logits = forward(*sample(64, True))
        rank = (logits > logits[:, :1]).sum(1)
        quality = dict(
            heldout_examples=64,
            hit_rate_at_10=float((rank < 10).float().mean()),
            candidate_count=k,
            random_baseline_hit_rate_at_10=10 / k,
            note="Small sampled-candidate smoke evaluation; not a production quality claim.",
        )
    np.savez(a.output / "hstu_weights.npz", **{k: v.detach().numpy() for k, v in weights.items()})
    provenance = dict(
        kind="public-data-trained-example",
        pretrained=False,
        seed=a.seed,
        training_steps=a.steps,
        batch_size=a.batch_size,
        architecture_source=UPSTREAM,
        architecture_revision=REVISION,
        dataset="MovieLens 100K",
        dataset_terms="https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        ratings_sha256=hashlib.sha256(a.ratings.read_bytes()).hexdigest(),
        sha256=hashlib.sha256((a.output / "hstu_weights.npz").read_bytes()).hexdigest(),
        initial_loss=losses[0],
        final_loss=losses[-1],
        evaluation=quality,
    )
    (a.output / "checkpoint_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
