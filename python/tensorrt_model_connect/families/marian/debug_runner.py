"""Marian-owned debug runner adapter."""

from __future__ import annotations

from tensorrt_model_connect.debug_runner import (
    Seq2SeqTrtRunner,
    load_vision_engine_from_bundle,
)


def runner_from_bundle(
    *,
    runtime_strategy: str,
    config: dict,
    header: dict,
    engine_plan: bytes,
    bundle_path: str,
    distributed_communicator: object | None = None,
) -> object | None:
    if runtime_strategy != "marian_translation":
        return None
    encoder_plan, _ = load_vision_engine_from_bundle(bundle_path)
    if encoder_plan is None:
        return None
    dec_layers = config.get("decoder_layers", header.get("num_layers", 1))
    decoder_start = config.get("decoder_start_token_id", 2)
    return Seq2SeqTrtRunner(
        decoder_plan=engine_plan,
        encoder_plan=encoder_plan,
        num_layers=dec_layers,
        max_cache_length=header["max_cache_length"],
        max_source_positions=header["max_cache_length"],
        decoder_start_token_id=decoder_start,
        distributed_communicator=distributed_communicator,
    )
