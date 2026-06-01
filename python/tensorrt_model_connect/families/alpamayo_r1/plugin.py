"""Alpamayo-R1 family plugin — NVIDIA autonomous-driving trajectory predictor.

Alpamayo-R1 (nvidia/Alpamayo-R1-{4B,10B}) combines:

  * A vision-language backbone (Qwen-VL-3, declared in config as
    ``vlm_backend: qwenvl3``) that ingests POV camera frames plus a
    structured driving-context prompt. All vlm.* weights mirror Qwen3-VL
    weight keys with a ``vlm.`` prefix added (verified against the actual
    model.safetensors.index.json — 749 weights under ``vlm.model.*`` plus
    ``vlm.lm_head.weight``).
  * A trajectory tokenizer (4000-token vocabulary) embedded into the LM
    head — the VLM autoregressively emits trajectory tokens after a
    ``traj_token_start_idx`` (151669) range. vocab_size=155697.
  * An action head (PerWaypointActionInProjV2 / Linear out) that decodes
    the trajectory tokens into 64 future waypoints in an
    UnicycleAccelCurvature action space (accel/curvature bounds).
  * A flow-matching diffusion module (FlowMatching with Euler integrator)
    that refines the waypoint predictions.

Onboarding blocker (verified during this scaffold work):

The released ``nvidia/Alpamayo-R1-10B`` checkpoint ships only:
  - config.json (Alpamayo-only fields: action_in_proj_cfg,
    action_space_cfg, diffusion_cfg, expert_cfg, traj_tokenizer_cfg,
    vocab_size=155697, vlm_backend="qwenvl3")
  - model.safetensors.index.json (1166 weight keys: vlm.* + expert.* +
    action_in_proj.* + action_out_proj.* + action_space.*)
  - The 5 safetensors shards.

It does **not** ship:
  - The Qwen3-VL backbone config (no num_hidden_layers / hidden_size /
    num_attention_heads at the VLM level — those are only in the
    Alpamayo expert config, which describes the trajectory expert, not
    the VLM).
  - A tokenizer (no tokenizer.json / tokenizer_config.json).
  - A processor (no preprocessor_config.json).
  - A vision encoder config (no visual/vision_config block).

Building the VLM portion requires pairing the Alpamayo weights with an
external Qwen3-VL-{4B,7B}-Instruct checkpoint as the config + tokenizer
source. The mapping (which Qwen3-VL variant matches which Alpamayo
checkpoint) is documented in the model card but not in config.json.

For now this plugin registers the family slot and raises a clear error
naming the missing pieces, so future onboarding work can pick up.

Once the supplemental config is provided, the loader path is:
  1. Use the helper below (``_wrap_readers_with_vlm_prefix``) to alias
     vlm.X → X so the Qwen3-VL loader sees standard keys.
  2. Call ``families.qwen_vl.plugin._load_qwen3_vl_weights`` with the
     wrapped readers (requires monkey-patching ``_open_safetensors`` on
     the qwen_vl plugin module since it imports the symbol at import time).
  3. Skip the expert.* / action_*.* / traj_tokenizer.* keys — they need
     separate engines + a custom C++ runtime path for waypoint
     reconstruction (flow-matching diffusion over 64 waypoints).
"""

from __future__ import annotations

from pathlib import Path

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict, _ReaderCollection, _open_safetensors


_VLM_PREFIX = "vlm."


def _wrap_readers_with_vlm_prefix(model_dir: Path) -> _ReaderCollection:
    """Open safetensors and add aliased entries that strip the ``vlm.`` prefix.

    Alpamayo-R1 stores its Qwen3-VL backbone under ``vlm.model.*`` and
    ``vlm.lm_head.*``. The qwen_vl loader looks for ``model.*`` /
    ``lm_head.*`` keys directly. This helper builds a ``_ReaderCollection``
    whose ``tensor_map`` exposes both the original (``vlm.X``) and the
    aliased (``X``) keys — the aliased entries route into a per-key proxy
    that rewrites the ``get_tensor`` call back to the original ``vlm.X``
    name. Documented here for future onboarding work.
    """

    readers = _open_safetensors(model_dir)
    aliased_tensor_map = dict(readers.tensor_map)
    for key, reader in list(readers.tensor_map.items()):
        if key.startswith(_VLM_PREFIX):
            stripped = key[len(_VLM_PREFIX):]
            if stripped not in aliased_tensor_map:
                aliased_tensor_map[stripped] = _PrefixedReader(reader, _VLM_PREFIX)
    return _ReaderCollection(list(readers), tensor_map=aliased_tensor_map)


class _PrefixedReader:
    """Proxy that prepends ``prefix`` before delegating get_tensor."""

    def __init__(self, inner, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix

    def get_tensor(self, name: str):
        return self._inner.get_tensor(self._prefix + name)

    def keys(self):
        for k in self._inner.keys():
            if k.startswith(self._prefix):
                yield k[len(self._prefix):]


_MISSING_SUPPLEMENTAL_MSG = (
    "alpamayo_r1: cannot build from the standalone nvidia/Alpamayo-R1-"
    "{4B,10B} checkpoint. The released repo only contains the Alpamayo "
    "orchestration config + weights; it lacks a Qwen3-VL backbone config "
    "(num_hidden_layers / hidden_size / num_attention_heads), tokenizer, "
    "processor, and vision_config. To build, pair the Alpamayo weights "
    "with the matching Qwen3-VL-{4B,7B}-Instruct checkpoint as the "
    "config + tokenizer source. See families/alpamayo_r1/plugin.py module "
    "docstring for the onboarding plan."
)


class AlpamayoR1Plugin:
    name = "alpamayo_r1"
    runtime_strategy = "vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "alpamayo_r1"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        raise NotImplementedError(_MISSING_SUPPLEMENTAL_MSG)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
    ) -> bytes:
        raise NotImplementedError(_MISSING_SUPPLEMENTAL_MSG)


plugin = AlpamayoR1Plugin()
