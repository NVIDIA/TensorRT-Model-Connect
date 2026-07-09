# Reference (DO-NOT-INTEGRATE) — MMMU-Pro VL evaluation harness

`mmmu_pro_eval.py` is the **reference** harness used for the Qwen3-VL quant/precision sweep.
Its `parse_multi_choice_response` is a **verbatim copy of the upstream MMMU-Pro parser**
(github.com/MMMU-Benchmark/MMMU  mmmu-pro/evaluate.py). This is intentionally the *official*
parser; it differs from `tools/task_eval.py`'s variant. Kept for reproducibility/reference only.
