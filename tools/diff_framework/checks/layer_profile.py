"""Per-layer TRT kernel timing check via IProfiler."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class LayerProfileTest:
    name = "layer_profile"
    description = "Per-layer TRT kernel timing via IProfiler (decoder models only)"
    runtime_strategies = []
    requires_bundle = False
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        import sys
        import time as _time
        from pathlib import Path

        # Ensure tools/ is on path so layer_profiler is importable
        tools_dir = Path(__file__).parent.parent.parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        t0 = _time.monotonic()
        try:
            from layer_profiler import LayerProfiler
            from perf_compare import build_trt_engine, load_trt_from_bundle
            from tensorrt_model_connect.engine_builder import _resolve_model
            from tool_helpers import make_family_debug_runner, runtime_strategy_from_config
            from transformers import AutoTokenizer
            import numpy as np

            model_dir = _resolve_model(ctx.model)
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, trust_remote_code=ctx.trust_remote_code)
            input_ids = tokenizer.encode("The capital of France is")
            eos_token_id = tokenizer.eos_token_id

            if ctx.bundle_path:
                engine_plan, num_layers, max_cache_length, bundle_config, _ = \
                    load_trt_from_bundle(ctx.bundle_path)
                runtime_strategy = str(bundle_config.get("runtime_strategy") or "")
                runner_config = bundle_config
                runner_bundle_path = ctx.bundle_path
            else:
                engine_plan, config, _, _ = build_trt_engine(
                    ctx.model, ctx.max_cache_length, ctx.verbose)
                num_layers = config.num_hidden_layers
                max_cache_length = ctx.max_cache_length
                runtime_strategy = runtime_strategy_from_config(config)
                runner_config = config
                runner_bundle_path = ""

            warmup, iterations, max_new_tokens = 1, 3, 10

            profiler = LayerProfiler()
            runner = make_family_debug_runner(
                engine_plan=engine_plan,
                runtime_strategy=runtime_strategy,
                max_cache_length=max_cache_length,
                num_layers=num_layers,
                config=runner_config,
                bundle_path=runner_bundle_path,
                profiler=profiler,
            )
            del engine_plan

            # Warmup then reset
            for _ in range(warmup):
                runner.reset()
                result = None
                for tid in input_ids:
                    result = runner.step(tid)
                logits = result["logits"].flatten()
                for _ in range(max_new_tokens):
                    next_token = int(np.argmax(logits))
                    if eos_token_id is not None and next_token == eos_token_id:
                        break
                    result = runner.step(next_token)
                    logits = result["logits"].flatten()
            profiler.reset()

            # Timed iterations
            for _ in range(iterations):
                runner.reset()
                result = None
                for tid in input_ids:
                    result = runner.step(tid)
                logits = result["logits"].flatten()
                for _ in range(max_new_tokens):
                    next_token = int(np.argmax(logits))
                    if eos_token_id is not None and next_token == eos_token_id:
                        break
                    result = runner.step(next_token)
                    logits = result["logits"].flatten()

            layer_data = profiler.to_dict()
            layers = layer_data.get("layers", [])
            total_ms = layer_data.get("total_ms", 0.0)

            top_layer = layers[0] if layers else {}
            metrics = {
                "total_layer_ms": round(total_ms, 3),
                "num_layers_profiled": len(layers),
                "bottleneck_layer": top_layer.get("name", ""),
                "bottleneck_pct": top_layer.get("pct", 0.0),
                "bottleneck_mean_ms": top_layer.get("mean_ms", 0.0),
            }

            msg = (f"{len(layers)} layers, total={total_ms:.2f}ms/step, "
                   f"bottleneck={top_layer.get('name', 'N/A')!r} "
                   f"({top_layer.get('pct', 0):.1f}%)")

            return DiffResult(
                test_name="layer_profile", model=ctx.model,
                runtime_strategy=ctx.runtime_strategy,
                passed=True,
                status="PASS",
                message=msg,
                metrics=metrics,
                duration_s=_time.monotonic() - t0,
            )
        except Exception as e:
            return DiffResult.error(
                "layer_profile", ctx.model, ctx.runtime_strategy, str(e))
