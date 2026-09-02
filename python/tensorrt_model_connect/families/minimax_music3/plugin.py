# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-Music3: lyrics and a music description to a full song.

The checkpoint is a seven-component pipeline declared by
``modular_model_index.json``::

    condition_encoder   diffusers    MiniMaxMusic3ConditionEncoder
    language_model      transformers Qwen3ForCausalLM
    rvq_depth_decoder   diffusers    MiniMaxMusic3RVQDepthDecoder
    scheduler           diffusers    FlowMatchEulerDiscreteScheduler
    tokenizer           transformers Qwen2Tokenizer
    transformer         diffusers    MiniMaxMusic3Transformer1DModel
    vocoder             diffusers    MiniMaxMusic3Vocoder

Upstream describes the weights as a Qwen3-8B fine-tune, a DiT-2B modified from
Stable Audio, and a VAE modified from the Descript Audio Codec.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

#: ``model_type`` in the published root ``config.json``.
SOURCE_MODEL_TYPE = "minimax_music3"

#: Root architecture entry in the published ``config.json``.
SOURCE_ARCHITECTURE = "MiniMaxMusic3ForConditionalGeneration"

#: Modular-pipeline class, present in ``modular_model_index.json`` only.
PIPELINE_CLASS = "MiniMaxMusic3ModularPipeline"

#: Components ``modular_model_index.json`` declares. Every one must be present
#: before this family claims a directory.
REQUIRED_COMPONENTS = (
    "condition_encoder",
    "language_model",
    "rvq_depth_decoder",
    "scheduler",
    "tokenizer",
    "transformer",
    "vocoder",
)

MODULAR_INDEX_NAME = "modular_model_index.json"


def read_pipeline_components(model_dir: Path) -> dict[str, tuple[str, str]]:
    """Return ``{component: (library, class)}`` from ``modular_model_index.json``.

    A conventional diffusers checkpoint would expose this through
    ``model_index.json``, which the shared resolver reads. MiniMax-Music3 ships
    only the modular index, so the family reads it itself.
    """

    index_path = Path(model_dir) / MODULAR_INDEX_NAME
    if not index_path.is_file():
        return {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(index, dict):
        return {}

    components: dict[str, tuple[str, str]] = {}
    for name, entry in index.items():
        if name.startswith("_") or not isinstance(entry, list) or len(entry) < 2:
            continue
        library, class_name = entry[0], entry[1]
        if isinstance(library, str) and isinstance(class_name, str):
            components[name] = (library, class_name)
    return components


class MiniMaxMusic3Plugin:
    """Builder plugin for the MiniMax-Music3 family.

    **Routing.** The published root ``config.json`` carries ``model_type`` and
    ``architectures`` but no ``_class_name``, and the repository ships no
    ``model_index.json``. ``_resolve_diffusion_entrypoint`` reads
    ``model_index.json`` first and otherwise requires ``_class_name`` in
    ``config.json``, so it declines this checkpoint and the
    ``diffusion_pipeline_classes`` route that ``minimax_h3`` uses is not
    available here. Routing therefore goes through the ordinary ``model_type``
    aliases, and the pipeline components are read from
    ``modular_model_index.json`` by :func:`read_pipeline_components`.

    **Task strategy.** ``text_to_audio``. The computation is a flow-matching
    diffusion loop, which would otherwise point at
    ``diffusion_media_generation``, and no existing family is both. The output
    decides it: the request shape this model needs is text in and a waveform
    out, and the ``tts_audio`` contract already transcribes generated audio and
    scores it against the input text, which is exactly the lyric-intelligibility
    check this model should be held to.

    **Input mapping.** The reference sends ``input`` and ``instructions``; the
    shared ``text_to_audio`` request carries one ``prompt``. The lyrics take
    ``prompt``, because the contract scores the transcript against it and
    folding the description into the same string would destroy that score. The
    description goes through the family's ``music_minimax_music3`` namespace.

    That keeps the shared contract untouched, but it does put several thousand
    characters of content through a channel whose other string fields hold a
    path or an enum. The alternative worth proposing to maintainers is a
    second, optional text input on the ``text_to_audio`` request itself, which
    is where a second *input* belongs; it is a shared-side change and so is not
    made here.
    """

    name = "minimax_music3"
    default_build_precision = "bf16"
    runtime_strategy = "minimax_music3_text_to_music"
    task_strategy = "text_to_audio"
    runtime_config_namespace = "music_minimax_music3"
    required_components = REQUIRED_COMPONENTS

    def matches(self, model_type: str) -> bool:
        """Claim the published ``model_type`` and its spelling variants."""

        return model_type.lower().replace("-", "_") in {
            "minimax_music3",
            "minimaxmusic3",
        }

    def matches_config(self, config: object) -> bool:
        """Claim a config that declares the MiniMax-Music3 architecture."""

        if isinstance(config, str):
            return self.matches(config)
        raw = getattr(config, "raw", None)
        if not isinstance(raw, dict):
            return False
        if raw.get("model_type") == SOURCE_MODEL_TYPE:
            return True
        architectures = raw.get("architectures")
        return (
            isinstance(architectures, list)
            and SOURCE_ARCHITECTURE in architectures
        )



    # -- build interface ---------------------------------------------------

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
        """Read every component's tensors from the snapshot.

        Three of the five have their full inventory recorded in
        :mod:`.checkpoint` and are checked against it; the transformer and the
        language model are sharded and are not enumerated there, so they are
        read without that check.
        """

        del config
        from . import engines
        from .checkpoint import validate_component

        root = Path(model_dir)
        missing = [
            component for component in REQUIRED_COMPONENTS
            if not (root / component).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "Incomplete MiniMax-Music3 checkpoint, missing: "
                + ", ".join(sorted(missing))
            )

        weights: dict = {"_model_dir": str(root)}
        for engine in engines.ENGINE_NAMES:
            component = engines.engine_component(engine)
            # The language model is the one component that must not be widened
            # at load: 36 layers plus an untied 200000 x 4096 head come to
            # roughly 35 GB at float32, and the build container is capped at
            # 125 GB, so widening it is killed by the OOM reaper before
            # TensorRT sees the weights. Its builder casts per tensor instead.
            tensors = _read_component(
                root / component, widen=engine != engines.LANGUAGE_MODEL_ENGINE)
            if component in _ENUMERATED_COMPONENTS:
                # The transformer and the language model are sharded and their
                # inventories are not enumerated in checkpoint.py; the other
                # three are, and are checked against the published headers.
                validate_component(component, {k: v.shape for k, v in tensors.items()})
            weights[engine] = tensors
        return weights

    def build_engine(self, config, weights, max_cache_length, **kwargs) -> bytes:
        """Build the diffusion transformer, the bundle's primary engine."""

        del config, max_cache_length
        from . import engines

        return _build_one(engines.DIT_ENGINE, weights, **kwargs)

    def build_extra_engines(self, config, weights, max_cache_length, **kwargs) -> dict:
        """Build the language model, depth decoder, condition encoder, vocoder."""

        del config
        from . import engines

        # Only the language model has a cache; the other three are feed-forward
        # over a fixed window, and _build_one ignores the argument for them.
        #
        # The caller's value is a floor, not a ceiling. This checkpoint's root
        # config.json carries none of the usual fields, so the builder reports
        # layers=0, hidden=0, vocab=0 and falls back to a 256-token default --
        # far short of the prompt plus every audio frame one generation emits.
        # An engine compiled for 256 would truncate every request.
        from .language_model_engine import DEFAULT_MAX_CACHE_LENGTH

        kwargs.setdefault(
            "max_cache_length", max(int(max_cache_length or 0), DEFAULT_MAX_CACHE_LENGTH)
        )
        # Free each component once its engine is serialised, and drop the
        # transformer's on the way in -- build_engine has already consumed it.
        #
        # This is not tidiness. Measured on an L40S container capped at 125 GB:
        # the language model's engine build alone peaks at 89 GB, and holding
        # the other four components (12 GB) plus the transformer plan the
        # caller still owns (9.3 GB) puts the total past the cap. The build
        # dies as 'Python builder terminated by signal 9'.
        weights.pop(engines.DIT_ENGINE, None)
        plans = {}
        for engine in engines.ENGINE_NAMES:
            if engine == engines.DIT_ENGINE:
                continue
            plans[f"{engine}_plan"] = _build_one(engine, weights, **kwargs)
            # The depth decoder reads the language model's embedding table, so
            # that component outlives its own engine by one step.
            if engine != engines.LANGUAGE_MODEL_ENGINE:
                weights.pop(engine, None)
            if engine == engines.DEPTH_DECODER_ENGINE:
                weights.pop(engines.LANGUAGE_MODEL_ENGINE, None)
            gc.collect()
        return plans

    def tokenizer_json_bundle_override(self, model_dir) -> bytes | None:
        """Return the tokenizer the runtime should carry.

        The builder looks for tokenizer.json beside config.json and, finding
        none, tries to convert a slow tokenizer and warns that it could not.
        This checkpoint is laid out as a modular pipeline, so its tokenizer is a
        component like any other: the real file is under ``tokenizer/`` and is
        already a fast-tokenizer serialisation. Point at it rather than
        regenerate what is published.
        """

        path = Path(model_dir) / "tokenizer" / "tokenizer.json"
        if not path.is_file():
            return None
        return path.read_bytes()

    def get_bundle_config_overrides(self, config) -> dict:
        """Return the runtime facts the checkpoint does not carry."""

        del config
        from . import engines

        return engines.bundle_config_overrides()


#: Components whose full tensor inventory :mod:`.checkpoint` records.
_ENUMERATED_COMPONENTS = frozenset(
    {"condition_encoder", "rvq_depth_decoder", "vocoder"}
)


def _read_component(directory: Path, *, widen: bool = True) -> dict:
    """Return one component's tensors as numpy arrays.

    With ``widen`` they come back float32, which is what the four feed-forward
    graphs want: the depth decoder is stored in bfloat16 and those builders
    assume float32, so widening at load is both necessary and lossless.

    The language model asks for ``widen=False``. Widening it does not fit in
    memory -- see the call site.

    When it happens, the widening is unconditional rather than a fallback
    from a failed numpy load. Whether ``safetensors.numpy`` can open a bfloat16 shard depends on
    whether ``ml_dtypes`` happens to be imported, which registers the dtype
    with numpy -- so a fallback keyed on the exception fires or does not fire
    depending on what else the process has imported. Loading then casting is
    the same result either way.
    """

    import numpy as np

    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {directory}")

    tensors: dict = {}
    for shard in shards:
        try:
            from safetensors.numpy import load_file

            loaded = load_file(str(shard))
        except TypeError:
            import torch
            from safetensors.torch import load_file as load_torch

            torch_tensors = load_torch(str(shard))
            if widen:
                loaded = {name: value.float().numpy()
                          for name, value in torch_tensors.items()}
            else:
                # .float() here would widen the very component that cannot
                # afford it, whatever the caller asked for. ml_dtypes gives
                # numpy a bfloat16 it understands, so the values cross over at
                # their stored width.
                import ml_dtypes

                loaded = {
                    name: value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
                    if value.dtype == torch.bfloat16 else value.numpy()
                    for name, value in torch_tensors.items()
                }
        if widen:
            tensors.update({
                name: np.ascontiguousarray(value, dtype=np.float32)
                for name, value in loaded.items()
            })
        else:
            tensors.update(loaded)
    return tensors


def _build_one(engine: str, weights: dict, *, latent_length: int = 689,
               steps: int = 8, max_cache_length: int | None = None,
               precision: str = "fp32", verbose: bool = False, **_kwargs) -> bytes:
    """Build one engine's serialized plan."""

    if max_cache_length is None:
        from .language_model_engine import DEFAULT_MAX_CACHE_LENGTH

        max_cache_length = DEFAULT_MAX_CACHE_LENGTH

    from tensorrt_model_connect import trt_compat

    from . import engines

    trt = trt_compat.get_trt()
    tensors = weights[engine]

    if engine == engines.LANGUAGE_MODEL_ENGINE:
        # The decoder builder assembles and serialises its own network: it owns
        # the cache bindings and the optimisation profile, which the hand-built
        # path here has no way to express.
        from . import language_model_engine

        # Precision matters most here. This engine carries a 200000 x 4096
        # embedding and an untied head of the same size, so fp32 costs about
        # 6.6 GB in those two tensors alone before the thirty-six layers.
        return language_model_engine.build_engine(
            tensors, max_cache_length=max_cache_length, precision=precision,
            verbose=verbose)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()

    if engine == engines.CONDITION_ENCODER_ENGINE:
        from . import condition_encoder_builder as builder_module
        from .pipeline_spec import CHUNK_FRAMES

        from .condition_encoder import engine_io_shapes

        source = network.add_input(
            builder_module.INPUT_NAME, trt.float32,
            tuple(engine_io_shapes(CHUNK_FRAMES)["hidden_states"]))
        output = builder_module.add_condition_encoder(
            network, trt, source, frames=CHUNK_FRAMES,
            mix=builder_module.folded_mix(
                tensors["layer_weight_logits"], tensors["layer_scale"]),
            proj_weight=tensors["proj.weight"], proj_bias=tensors["proj.bias"])
        output.name = builder_module.OUTPUT_NAME
    elif engine == engines.DEPTH_DECODER_ENGINE:
        from . import depth_decoder_builder as builder_module
        from . import depth_decoder_engine as depth_engine
        from .depth_decoder import HIDDEN_SIZE, NUM_CODEBOOKS

        lm_hidden = network.add_input(depth_engine.HIDDEN_INPUT, trt.float32,
                                      (1, 1, HIDDEN_SIZE))
        # Eight codes in: the seven the sequence uses, plus the last draw, which
        # only the frame embedding reads.
        codes = network.add_input(depth_engine.CODES_INPUT, trt.int32,
                                  (1, NUM_CODEBOOKS))
        # The depth sequence's second position is the language model's own
        # embedding of the semantic code, so this engine needs that table.
        language_model_tensors = weights.get(engines.LANGUAGE_MODEL_ENGINE)
        if language_model_tensors is None:
            raise ValueError(
                "the depth decoder engine needs the language model's embeddings; "
                "build it before the language model's weights are released"
            )
        sequence = depth_engine.add_input_sequence(
            network, trt, lm_hidden, codes, weights=tensors,
            embed_tokens=language_model_tensors["model.embed_tokens.weight"])
        hidden = builder_module.add_depth_decoder(
            network, trt, sequence, steps=depth_engine.MAX_STEPS, weights=tensors)
        output = depth_engine.add_output_heads(
            network, trt, hidden, weights=tensors)
        output.name = depth_engine.LOGITS_OUTPUT
        # The condition encoder reads these seven states alongside the language
        # model's one; without them the encoder has nothing to weight.
        depth_hidden = depth_engine.add_hidden_output(network, trt, hidden)
        depth_hidden.name = depth_engine.HIDDEN_OUTPUT
        network.mark_output(depth_hidden)
        frame_embed = depth_engine.add_frame_embedding(
            network, trt, codes, weights=tensors,
            embed_tokens=language_model_tensors["model.embed_tokens.weight"])
        frame_embed.name = depth_engine.FRAME_EMBED_OUTPUT
        network.mark_output(frame_embed)
    elif engine == engines.VOCODER_ENGINE:
        from . import vocoder_builder as builder_module

        source = network.add_input(
            builder_module.INPUT_NAME, trt.float32,
            builder_module.expected_input_shape(latent_length))
        output = builder_module.add_vocoder(
            network, trt, source, latent_length=latent_length, weights=tensors)
        output.name = builder_module.OUTPUT_NAME
    elif engine == engines.DIT_ENGINE:
        from . import dit_builder as builder_module
        from .dit import CONDITION_DIM, IN_CHANNELS

        latents = network.add_input(builder_module.LATENTS_NAME, trt.float32,
                                    (1, IN_CHANNELS, latent_length))
        condition = network.add_input(builder_module.CONDITION_NAME, trt.float32,
                                      (1, latent_length, CONDITION_DIM))
        # The engine takes the flow-matching time as a scalar and embeds it
        # itself. Exposing the already-embedded prefix instead would put
        # time_proj and the two time_embed layers outside the bundle, leaving
        # the runtime to reproduce them by hand.
        timestep = network.add_input(builder_module.TIMESTEP_SCALAR_NAME,
                                     trt.float32, (1, 1, 1))
        embedded = builder_module.add_timestep_embedding(
            network, trt, timestep, weights=tensors)
        output = builder_module.add_dit(
            network, trt, latents, condition, embedded,
            latent_length=latent_length, weights=tensors)
        output.name = builder_module.OUTPUT_NAME
    else:
        raise ValueError(f"unknown MiniMax-Music3 engine {engine!r}")

    network.mark_output(output)
    builder_module.configure(config, trt)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"MiniMax-Music3 {engine} engine build returned None")
    return bytes(plan)


plugin = MiniMaxMusic3Plugin()
