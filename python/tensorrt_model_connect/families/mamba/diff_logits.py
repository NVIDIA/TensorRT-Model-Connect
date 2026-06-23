"""Logit diff hooks owned by the Mamba family."""

from __future__ import annotations


def handles_model_type(model_type: str) -> bool:
    return model_type.lower() == "mamba"


def make_trt_runner(engine_plan, config, max_cache_length):
    del max_cache_length
    from tensorrt_model_connect.families.mamba.debug_runner import MambaTrtRunner

    return MambaTrtRunner(
        engine_plan=engine_plan,
        num_layers=config.num_hidden_layers,
    )
