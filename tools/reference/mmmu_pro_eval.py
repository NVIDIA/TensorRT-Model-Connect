import argparse, ast, glob, json, random, re, subprocess, sys

random.seed(42)  # deterministic random fallback in the MMMU-Pro parser (matches official)

def build_prompt(q, opts):
    q = re.sub(r'<image \d+>', '', q).strip()
    letters = [chr(65 + i) for i in range(len(opts))]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
    # NOTE: this is the official MMMU **Direct-Standard** prompt. For the industry
    # CoT protocol (future, requires re-run) switch to the MMMU-Pro CoT prompt:
    #   "Answer the preceding multiple choice question. The last line of your response
    #    should be of the following format: 'Answer: $LETTER' ... Think step by step ..."
    return f"{q}\n{body}\nAnswer with the option's letter from the given choices directly."


# =========================================================================
# ANSWER PARSER — verbatim MMMU-Pro parse_multi_choice_response (2026-07-07).
# Source: github.com/MMMU-Benchmark/MMMU  mmmu-pro/evaluate.py (CoT-aware:
# last "Answer:" -> letter; else (A)/A /A. ; else option-content match; else
# random.choice, seed 42). Verbatim copy in mmmu_official_reference/evaluate.py.
# Always returns a letter (random fallback) => no None. GARBAGE (!!!!) is
# counted SEPARATELY (see is_garbage) so the MC-only decode bug is never hidden
# by the parser's random fallback.
# Our previous custom parser is preserved below as parse_letter_legacy (unused).
# =========================================================================
def parse_multi_choice_response(response, all_choices, index2ans):
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        matching = [o for o in all_choices if o in answer_str]
        if len(matching) == 1:
            return matching[0]
    if isinstance(response, str):
        for ch in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(ch)
        response = " " + response + " "
    else:
        response = ""
    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:            # (A) (B) ...
        if f"({choice})" in response:
            candidates.append(choice); ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:        # "A " "B " ...
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:        # "A." "B." ...
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:   # option-content match
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index); index_ans = False
    if len(candidates) == 0:
        return random.choice(all_choices)                    # fallback
    if len(candidates) > 1:
        starts = []
        for can in candidates:
            if index_ans:
                key = f"({can})" if ans_with_brack else f" {can} "
                starts.append(response.rfind(key))
            else:
                starts.append(response.lower().rfind(index2ans[can].lower()))
        return candidates[max(range(len(starts)), key=lambda k: starts[k])]
    return candidates[0]

def is_garbage(gen):
    """MC-only decode bug: output is pure '!' (repeated) — flagged, not hidden."""
    g = (gen or "").strip()
    return bool(g) and set(g) <= set("!")

def parse_pred(gen, opts):
    all_choices = [chr(65 + k) for k in range(len(opts))]
    index2ans = {chr(65 + k): opts[k] for k in range(len(opts))}
    return parse_multi_choice_response(gen, all_choices, index2ans)

def parse_letter_legacy(text, n):   # PRE-2026-07-07 custom parser (kept for reference/revert)
    last = chr(64 + n)
    t = text.strip()
    for pat in [r'answer\s*(?:is|:)?\s*\(?([A-' + last + r'])\)?',
                r'\b([A-' + last + r'])\s*[.):]',
                r'\(([A-' + last + r'])\)',
                r'option\s+([A-' + last + r'])']:
        m = re.findall(pat, t, re.IGNORECASE)
        if m:
            return m[-1].upper()
    m = re.findall(r'\b([A-' + last + r'])\b', t)
    return m[-1].upper() if m else None

def load_samples(data, limit):
    from datasets import load_from_disk
    ds = load_from_disk(data)
    out = []
    for r in ds:
        if r.get('image_2') is not None:            # single-image only (MC constraint)
            continue
        if r.get('question_type') != 'multiple-choice':
            continue
        try:
            opts = ast.literal_eval(r['options'])
        except Exception:
            continue
        if not opts or len(opts) < 2:
            continue
        out.append((r['question'], opts, r['answer'], r['image_1']))
        if limit and len(out) >= limit:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True)          # hf | mc
    ap.add_argument('--limit', type=int, default=150)
    ap.add_argument('--bundle', default=None)            # for mc
    ap.add_argument('--data', default='/build/mmmu/validation')
    ap.add_argument('--quantize', default=None)          # hf fake-quant: fp8 | int8 | None
    ap.add_argument('--dtype', default='bf16')           # hf base dtype: bf16 | fp16
    ap.add_argument('--calib-n', type=int, default=64)   # hf fake-quant: image+text calib sample count
    ap.add_argument('--out', default=None)               # save raw generations jsonl (offline re-parse)
    ap.add_argument('--max-new-tokens', type=int, default=256)
    ap.add_argument('--greedy', action='store_true')   # MC: match HF do_sample=False (deterministic argmax)
    a = ap.parse_args()
    S = load_samples(a.data, a.limit)
    tag = f"{a.backend}/{a.dtype}" + (f"+{a.quantize}" if a.quantize else "")
    print(f"MMMU eval: {len(S)} single-image MC samples | {tag}", flush=True)
    correct = 0; scored = 0; garbage = 0
    outf = open(a.out, 'w') if a.out else None

    if a.backend == 'hf':
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        mp = glob.glob('/build/hf/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/')[0]
        ct = json.load(open(mp + 'tokenizer_config.json')).get('chat_template')
        proc = AutoProcessor.from_pretrained(mp)
        dt = {'bf16': torch.bfloat16, 'fp16': torch.float16}[a.dtype]
        model = AutoModelForImageTextToText.from_pretrained(mp, torch_dtype=dt, device_map='cuda').eval()

        def make_inp(q, opts, img):
            msgs = [{"role": "user", "content": [{"type": "image", "image": img.convert('RGB')},
                                                 {"type": "text", "text": build_prompt(q, opts)}]}]
            return proc.apply_chat_template(msgs, chat_template=ct, add_generation_prompt=True,
                                            tokenize=True, return_dict=True, return_tensors="pt").to('cuda')

        if a.quantize:
            # Fake-quant the torch reference with the SAME ModelOpt config MC uses,
            # and the SAME scope (body only: exclude lm_head / vision / norms / embed).
            import modelopt.torch.quantization as mtq
            cfgname = {'fp8': 'FP8_DEFAULT_CFG', 'int8': 'INT8_SMOOTHQUANT_CFG', 'nvfp4': 'NVFP4_DEFAULT_CFG'}[a.quantize]
            cfg = getattr(mtq, cfgname)
            calib = S[:a.calib_n]  # image+text, range-observation only (no label leakage)
            def fwd(m):
                for (q, opts, ans, img) in calib:
                    with torch.no_grad():
                        m(**make_inp(q, opts, img))
            print(f"[hf-quant] calibrating {cfgname} on {len(calib)} samples ...", flush=True)
            mtq.quantize(model, cfg, fwd)
            exc = re.compile(r'(lm_head|visual|vision|embed|norm)')
            ndis = 0
            for name, mod in model.named_modules():
                if exc.search(name):
                    for qz in ("input_quantizer", "weight_quantizer"):
                        try:
                            mtq.disable_quantizer(mod, qz); ndis += 1
                        except Exception:
                            pass
            print(f"[hf-quant] {cfgname} applied; disabled {ndis} excluded quantizers (body-only scope)", flush=True)

        for i, (q, opts, ans, img) in enumerate(S):
            inp = make_inp(q, opts, img)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
            gen = proc.decode(o[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = parse_pred(gen, opts); scored += 1; correct += (pred == ans); garbage += is_garbage(gen)
            if outf: outf.write(json.dumps({'i': i, 'ans': ans, 'pred': pred, 'gen': gen}) + '\n'); outf.flush()
            if i < 5 or (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(S)}] ans={ans} pred={pred} :: {gen[:40]!r}", flush=True)

    elif a.backend == 'mc':
        T = '/build/trtmc-build-trt11/trtmc'
        for i, (q, opts, ans, img) in enumerate(S):
            img.convert('RGB').save('/tmp/mmmu.png')
            prompt = build_prompt(q, opts)
            cmd = [T, 'run', a.bundle, '--image', '/tmp/mmmu.png', '--prompt', prompt,
                   '--max-new-tokens', str(a.max_new_tokens), '--chat-template']
            if a.greedy: cmd.append('--greedy')   # deterministic argmax, matches HF do_sample=False
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=180)
            gen = '\n'.join(l for l in p.stdout.splitlines()
                            if not l.startswith('[trtmc') and 'trtmc]' not in l).strip()
            pred = parse_pred(gen, opts); scored += 1; correct += (pred == ans); garbage += is_garbage(gen)
            if outf: outf.write(json.dumps({'i': i, 'ans': ans, 'pred': pred, 'gen': gen}) + '\n'); outf.flush()
            if i < 5 or (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(S)}] ans={ans} pred={pred} :: {gen[:40]!r}", flush=True)

    if outf: outf.close()
    print(f"{tag} MMMU accuracy: {correct}/{scored} = {100*correct/max(scored,1):.1f}%"
          f"  (parser=MMMU-Pro; garbage(!!!!)={garbage} = MC-only decode bug, counted not hidden)", flush=True)


if __name__ == '__main__':
    main()
