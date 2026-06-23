from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-hf-reference", action="store_true")
    parser.add_argument("--dense-bundle")
    parser.add_argument("--tri-bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=38912)
    parser.add_argument("--dense-gpu", default="1")
    parser.add_argument("--tri-gpu", default="2")
    parser.add_argument("--hf-gpu", default="3")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--hf-attn-impl", default="eager")
    parser.add_argument("--hf-dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    parser.add_argument("--hf-python", default=sys.executable)
    parser.add_argument("--trtmc-benchmark-binary", default="build/trtmc_dataset_benchmark")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hf-shard-index", type=int, default=0)
    parser.add_argument("--hf-num-shards", type=int, default=1)
    parser.add_argument("--hf-output-path")
    parser.add_argument(
        "--dense-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env override for the dense TRT run (repeatable)",
    )
    parser.add_argument(
        "--tri-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env override for the TriAttention TRT run (repeatable)",
    )
    # Generic config surface. No per-knob flags. Forwards to the trtmc
    # binary via --config / --set; dense- and tri-specific setters layer on
    # top of the shared --config for asymmetric runs.
    parser.add_argument(
        "--config", default=None, metavar="FILE",
        help="Shared config profile (.json). Applied to both dense and tri runs.",
    )
    parser.add_argument(
        "--set", action="append", default=[], dest="shared_set", metavar="NS.FIELD=VALUE",
        help="Shared session-layer override, applied to both dense and tri runs (repeatable).",
    )
    parser.add_argument(
        "--dense-set", action="append", default=[], metavar="NS.FIELD=VALUE",
        help="Session-layer override applied only to the dense run (repeatable).",
    )
    parser.add_argument(
        "--tri-set", action="append", default=[], metavar="NS.FIELD=VALUE",
        help="Session-layer override applied only to the tri run (repeatable).",
    )
    args = parser.parse_args()
    if not args.run_hf_reference and (not args.dense_bundle or not args.tri_bundle):
        parser.error("--dense-bundle and --tri-bundle are required unless --run-hf-reference is set")
    if args.hf_num_shards < 1:
        parser.error("--hf-num-shards must be >= 1")
    if args.hf_shard_index < 0 or args.hf_shard_index >= args.hf_num_shards:
        parser.error("--hf-shard-index must satisfy 0 <= shard < num_shards")
    return args


def parse_extra_env(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid env override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid env override {item!r}; empty key")
        env[key] = value
    return env


def write_dataset(output_dir: Path, limit: int) -> Path:
    from datasets import load_dataset

    ds = load_dataset("MathArena/aime_2025", split="train")
    path = output_dir / "aime25_prompts.jsonl"
    with path.open("w", encoding="utf-8") as f:
        count = 0
        for row in ds:
            prompt = PROMPT_TEMPLATE.format(question=row["problem"])
            sample = {
                "dataset_index": count,
                "sample_id": f"aime25_{row['problem_idx']}",
                "answer": str(row["answer"]),
                "question": row["problem"],
                "prompt": prompt,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if limit > 0 and count >= limit:
                break
    return path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_gpu_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shard_rows(rows: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards == 1:
        return rows
    return rows[shard_index::num_shards]


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
FINAL_RE = re.compile(r"Final answer:\s*([^\n\r]+)")
INT_RE = re.compile(r"-?\d+")
ANSWER_RE = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE)
DISCOURSE_QUANTITY_RE = re.compile(
    r"(?:therefore|thus|hence|so),?\s+"
    r"(?:the\s+)?(?:answer|area|sum|difference|product|remainder|probability|count|number|value|total)"
    r"[^\n\r]{0,64}?(?:is|=)\s*(-?\d+)",
    re.IGNORECASE,
)
MN_RE = re.compile(r"m\s*\+\s*n\s*=\s*(-?\d+)", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    boxed = BOXED_RE.search(text)
    if boxed:
        return normalize_answer(boxed.group(1))
    final = FINAL_RE.search(text)
    if final:
        return normalize_answer(final.group(1))
    phrase_match = None
    for pattern in (ANSWER_RE, DISCOURSE_QUANTITY_RE, MN_RE):
        for match in pattern.finditer(text):
            phrase_match = normalize_answer(match.group(1))
    if phrase_match is not None:
        return phrase_match
    ints = INT_RE.findall(text)
    if ints:
        return ints[-1].lstrip("+")
    return None


def normalize_answer(value: str) -> str | None:
    ints = INT_RE.findall(value.replace(",", ""))
    if not ints:
        return None
    try:
        return str(int(ints[0]))
    except ValueError:
        return ints[0].lstrip("+")


def aggregate(rows: list[dict], hf_by_id: dict[str, dict] | None = None) -> dict:
    total_tokens = 0
    total_wall_ms = 0.0
    total_decode_ms = 0.0
    correct = 0
    hf_match = 0
    samples: list[dict] = []
    for row in rows:
        pred = normalize_answer(str(row.get("pred_answer", ""))) if row.get("pred_answer") else extract_answer(row["text"])
        gold = normalize_answer(str(row.get("gold_answer", row.get("answer"))))
        is_correct = pred == gold and pred is not None
        correct += int(is_correct)
        hf_pred = None
        hf_agree = None
        if hf_by_id is not None:
            hf_row = hf_by_id[row["sample_id"]]
            hf_pred = (
                normalize_answer(str(hf_row.get("pred_answer", "")))
                if hf_row.get("pred_answer")
                else extract_answer(hf_row["text"])
            )
            hf_agree = pred == hf_pred and pred is not None and hf_pred is not None
            hf_match += int(bool(hf_agree))
        total_tokens += int(row["generated_tokens"])
        total_wall_ms += float(row["wall_ms"])
        total_decode_ms += float(row.get("decode_ms", 0.0))
        samples.append(
            {
                "sample_id": row["sample_id"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "hf_pred": hf_pred,
                "hf_match": hf_agree,
                "generated_tokens": row["generated_tokens"],
                "wall_ms": row["wall_ms"],
                "decode_ms": row.get("decode_ms"),
            }
        )

    count = len(rows)
    return {
        "count": count,
        "accuracy": (correct / count) if count else 0.0,
        "hf_agreement": (hf_match / count) if hf_by_id is not None and count else None,
        "total_generated_tokens": total_tokens,
        "total_wall_ms": total_wall_ms,
        "total_decode_ms": total_decode_ms,
        "wall_tokens_per_sec": (total_tokens / (total_wall_ms / 1000.0)) if total_wall_ms > 0 else 0.0,
        "decode_tokens_per_sec": (
            total_tokens / (total_decode_ms / 1000.0) if total_decode_ms > 0 else None
        ),
        "samples": samples,
    }


def run_subprocess(cmd: list[str], *, env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def run_hf_reference(
    *,
    dataset_path: Path,
    output_path: Path,
    model_id: str,
    attn_impl: str,
    hf_dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    seed: int,
    use_chat_template: bool,
    enable_thinking: bool,
    shard_index: int,
    num_shards: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class AnswerStoppingCriteria(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length: int, check_every: int = 16):
            super().__init__()
            self.tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.check_every = max(check_every, 1)

        def __call__(self, input_ids, scores, **kwargs): # type: ignore[override]
            generated = input_ids[0, self.prompt_length:]
            if generated.numel() == 0 or (generated.numel() % self.check_every) != 0:
                return False
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return BOXED_RE.search(text) is not None or FINAL_RE.search(text) is not None

    rows = shard_rows(load_jsonl(dataset_path), shard_index, num_shards)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype: str | torch.dtype
    if hf_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif hf_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    ).eval().cuda()

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = row["prompt"]
            if use_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            if seed >= 0:
                sample_seed = seed + int(row.get("dataset_index", 0))
                torch.manual_seed(sample_seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            stopping_criteria = StoppingCriteriaList(
                [AnswerStoppingCriteria(tokenizer, int(inputs["input_ids"].shape[1]))]
            )
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
            generated = outputs[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            generated_tokens = int(generated.shape[0])
            row_out = {
                "sample_id": row["sample_id"],
                "gold_answer": row["answer"],
                "pred_answer": extract_answer(text),
                "generated_tokens": generated_tokens,
                "wall_ms": wall_ms,
                "tokens_per_sec": generated_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
                "text": text,
            }
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[hf.reference] sample={row['sample_id']} generated_tokens={generated_tokens} "
                f"wall_ms={wall_ms:.3f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = write_dataset(output_dir, args.limit)

    dense_out = output_dir / "dense_results.jsonl"
    tri_out = output_dir / "tri_results.jsonl"
    hf_gpus = parse_gpu_list(args.hf_gpu)
    hf_outputs = (
        [Path(args.hf_output_path)]
        if args.run_hf_reference and args.hf_output_path
        else [output_dir / "hf_results.jsonl"]
        if len(hf_gpus) == 1
        else [output_dir / f"hf_results.shard{i}.jsonl" for i in range(len(hf_gpus))]
    )

    dense_log = output_dir / "dense_benchmark.log"
    tri_log = output_dir / "tri_benchmark.log"
    hf_logs = (
        [output_dir / "hf_reference.log"]
        if len(hf_outputs) == 1
        else [output_dir / f"hf_reference.shard{i}.log" for i in range(len(hf_outputs))]
    )

    base_env = os.environ.copy()
    use_chat_template = not args.no_chat_template
    enable_thinking = not args.no_thinking

    dense_env = base_env | {"CUDA_VISIBLE_DEVICES": args.dense_gpu}
    tri_env = base_env | {
        "CUDA_VISIBLE_DEVICES": args.tri_gpu,
    }
    dense_env |= parse_extra_env(args.dense_env)
    tri_env |= parse_extra_env(args.tri_env)
    hf_envs = [base_env | {"CUDA_VISIBLE_DEVICES": gpu} for gpu in hf_gpus]

    dense_cmd = [
        args.trtmc_benchmark_binary,
        args.dense_bundle,
        str(dataset_path),
        str(dense_out),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--seed",
        str(args.seed),
        "--stop-on-answer",
    ]
    tri_cmd = [
        args.trtmc_benchmark_binary,
        args.tri_bundle,
        str(dataset_path),
        str(tri_out),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
        "--seed",
        str(args.seed),
        "--stop-on-answer",
    ]
    if use_chat_template:
        dense_cmd.append("--chat-template")
        tri_cmd.append("--chat-template")
    if not enable_thinking:
        dense_cmd.append("--no-thinking")
        tri_cmd.append("--no-thinking")

    # Generic config surface: --config applies to both runs; per-run --set
    # overlays on top. No per-knob flags — new runtime knobs flow through
    # here without any edit to the benchmark.
    if args.config:
        dense_cmd.extend(["--config", args.config])
        tri_cmd.extend(["--config", args.config])
    dense_defaults = ["runtime.prefer_gpu_greedy=true"]
    tri_defaults = ["runtime.prefer_gpu_greedy=true", "triattention.profile=true"]
    for token in dense_defaults + args.shared_set + args.dense_set:
        dense_cmd.extend(["--set", token])
    for token in tri_defaults + args.shared_set + args.tri_set:
        tri_cmd.extend(["--set", token])
    if args.run_hf_reference:
        run_hf_reference(
            dataset_path=dataset_path,
            output_path=hf_outputs[0],
            model_id=args.model_id,
            attn_impl=args.hf_attn_impl,
            hf_dtype=args.hf_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            seed=args.seed,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            shard_index=args.hf_shard_index,
            num_shards=args.hf_num_shards,
        )
        return

    dense_proc = run_subprocess(dense_cmd, env=dense_env, log_path=dense_log)
    tri_proc = run_subprocess(tri_cmd, env=tri_env, log_path=tri_log)
    hf_procs = []
    for i, (hf_env, hf_out, hf_log) in enumerate(zip(hf_envs, hf_outputs, hf_logs, strict=True)):
        hf_cmd = [
            args.hf_python,
            __file__,
            "--run-hf-reference",
            "--output-dir",
            str(output_dir),
            "--model-id",
            args.model_id,
            "--hf-attn-impl",
            args.hf_attn_impl,
            "--hf-dtype",
            args.hf_dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--min-p",
            str(args.min_p),
            "--seed",
            str(args.seed),
            "--limit",
            str(args.limit),
            "--hf-shard-index",
            str(i),
            "--hf-num-shards",
            str(len(hf_gpus)),
            "--hf-output-path",
            str(hf_out),
        ]
        if not use_chat_template:
            hf_cmd.append("--no-chat-template")
        if not enable_thinking:
            hf_cmd.append("--no-thinking")
        hf_procs.append(run_subprocess(hf_cmd, env=hf_env, log_path=hf_log))

    codes = {
        "dense": dense_proc.wait(),
        "tri": tri_proc.wait(),
    }
    for i, proc in enumerate(hf_procs):
        codes[f"hf[{i}]"] = proc.wait()
    failed = {name: code for name, code in codes.items() if code != 0}
    if failed:
        raise SystemExit(f"Benchmark jobs failed: {failed}")

    dense_rows = load_jsonl(dense_out)
    tri_rows = load_jsonl(tri_out)
    hf_rows: list[dict] = []
    for path in hf_outputs:
        hf_rows.extend(load_jsonl(path))
    hf_by_id = {row["sample_id"]: row for row in hf_rows}

    summary = {
        "dataset": "aime25",
        "count": len(hf_rows),
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "seed": args.seed,
            "chat_template": use_chat_template,
            "enable_thinking": enable_thinking,
            "hf_dtype": args.hf_dtype,
            "stop_on_answer": True,
        },
        "dense": aggregate(dense_rows, hf_by_id),
        "tri": aggregate(tri_rows, hf_by_id),
        "hf": aggregate(hf_rows, None),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = output_dir / "summary.md"
    md.write_text(
        "\n".join(
            [
                "# Qwen3-8B AIME25 Benchmark",
                "",
                f"- max_new_tokens: {args.max_new_tokens}",
                f"- count: {summary['count']}",
                (
                    f"- sampling: temp={args.temperature}, top_k={args.top_k}, "
                    f"top_p={args.top_p}, min_p={args.min_p}, seed={args.seed}"
                ),
                f"- chat_template: {use_chat_template}",
                f"- enable_thinking: {enable_thinking}",
                f"- hf_dtype: {args.hf_dtype}",
                "",
                "| Method | Accuracy | HF Agreement | Wall tok/s | Decode tok/s |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| dense | {summary['dense']['accuracy']:.3f} | "
                    f"{summary['dense']['hf_agreement']:.3f} | "
                    f"{summary['dense']['wall_tokens_per_sec']:.2f} | "
                    f"{summary['dense']['decode_tokens_per_sec']:.2f} |"
                ),
                (
                    f"| tri | {summary['tri']['accuracy']:.3f} | "
                    f"{summary['tri']['hf_agreement']:.3f} | "
                    f"{summary['tri']['wall_tokens_per_sec']:.2f} | "
                    f"{summary['tri']['decode_tokens_per_sec']:.2f} |"
                ),
                (
                    f"| hf-eager | {summary['hf']['accuracy']:.3f} | "
                    f"n/a | {summary['hf']['wall_tokens_per_sec']:.2f} | n/a |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
