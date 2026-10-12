# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect model-card outputs from Cloudflare's pinned original implementation."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import torch

from families.clef.tests.compare_head import dump_inputs, head_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parent / "fixtures")
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkpoint))
    from joint_schema_model import (
        collate_records,
        encode_record,
        load_release_model,
        systemone_answer,
    )

    print("Loading pinned original model", flush=True)
    model, processor = load_release_model(args.checkpoint, device="cpu")
    model = model.to("cuda")
    receipts = []
    with torch.inference_mode():
        for fixture in sorted(args.fixtures.glob("*.json")):
            from families.clef.tests.media_fixtures import fixture_record

            record = fixture_record(fixture)
            encoded = encode_record(processor.tokenizer, record, processor=processor)
            batch = collate_records(
                [encoded], processor.tokenizer.pad_token_id, torch.device("cuda")
            )
            print("Reference forward:", fixture.stem, len(encoded.input_ids), flush=True)
            backbone = (
                model.language_model.model
                if batch["media"]
                else model.language_model.model.language_model
            )
            hidden = backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
                **batch["media"],
            ).last_hidden_state
            embedding = model.language_model.get_output_embeddings().weight
            logits = model.head(
                hidden, batch["input_ids"], batch["attention_mask"], batch["records"], embedding
            )[0]
            answers, scores = {}, []
            for question, values in zip(encoded.questions, logits):
                probs = values.float().softmax(-1).tolist()
                answers[question.question_id] = systemone_answer(
                    record["questions"][question.question_id], dict(zip(question.option_ids, probs))
                )
                scores.append(
                    {
                        "question_id": question.question_id,
                        "option_ids": question.option_ids,
                        "logits": values.float().tolist(),
                        "probabilities": probs,
                    }
                )
            root = args.output / fixture.stem
            dump_inputs(root, head_inputs(hidden[0], embedding, encoded))
            torch.cat(logits).float().cpu().numpy().tofile(root / "logits.reference.bin")
            encoded_data = dataclasses.asdict(encoded)
            encoded_data.pop("media", None)
            (root / "encoded.json").write_text(json.dumps(encoded_data))
            output = {
                "model": "clef",
                "answers": answers,
                "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
            }
            (root / "reference.json").write_text(
                json.dumps({"response": output, "scores": scores}, indent=2)
            )
            print(json.dumps(output), flush=True)
            receipts.append({"fixture": fixture.stem, "response": output, "scores": scores})
    (args.output / "reference.json").write_text(
        json.dumps(
            {
                "checkpoint_revision": "2f3de3dd85f379784083b0814d997ab627200f0c",
                "torch": torch.__version__,
                "results": receipts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
