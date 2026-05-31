"""Alpamayo-R1 family plugin — NVIDIA autonomous-driving trajectory predictor.

Alpamayo-R1 (nvidia/Alpamayo-R1-{4B,10B}) is a domain-specific model for
autonomous driving. It combines:

  * A vision-language backbone (Qwen-VL-3, declared in config as
    ``vlm_backend: qwenvl3``) that ingests the driver's POV camera frames
    plus a structured driving-context prompt.
  * A trajectory tokenizer (4000-token vocabulary) embedded into the LM
    head — the VLM autoregressively emits trajectory tokens after a
    ``traj_token_start_idx`` (151669) range.
  * An action head (PerWaypointActionInProjV2 / Linear out) that decodes
    the trajectory tokens into 64 future waypoints in an
    UnicycleAccelCurvature action space (accel/curvature bounds).
  * A flow-matching diffusion module (FlowMatching with Euler integrator)
    that refines the waypoint predictions.

The model isn't a standard causal LM and doesn't fit the existing
decoder_kv_cache / vision_language runtime strategies — it needs a new
``trajectory_action`` runtime strategy with:

  1. A custom weight loader that reuses the existing Qwen-VL-3 backbone
     load path (via families/qwen_vl) and adds the action_in_proj /
     action_out_proj / traj_tokenizer / flow-matching weights.
  2. A new builder that produces both the VLM TRT engine (reusing the
     qwen_vl/decoder_tp_builder) AND a separate flow-matching diffusion
     engine for the action refinement step.
  3. A new C++ runtime path that runs the VLM prefill, samples
     trajectory tokens, then runs flow-matching steps over the candidate
     waypoints.

This stub plugin matches the ``alpamayo_r1`` model_type so the harness
emits a clear NotImplementedError instead of the generic "No supported
build backend matched this model" — and so future onboarding work has
a registered family slot to fill in.
"""

from __future__ import annotations

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict


class AlpamayoR1Plugin:
    name = "alpamayo_r1"
    runtime_strategy = "trajectory_action"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "alpamayo_r1"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        raise NotImplementedError(
            "alpamayo_r1 family: weight loading not implemented yet. "
            "The model has a Qwen-VL-3 backbone (delegate to qwen_vl) plus "
            "action_in_proj / action_out_proj / traj_tokenizer / flow-matching "
            "weights that need custom mapping. See families/alpamayo_r1/plugin.py "
            "module docstring for the onboarding outline."
        )

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        raise NotImplementedError(
            "alpamayo_r1 family: engine build not implemented yet. "
            "See families/alpamayo_r1/plugin.py module docstring."
        )


plugin = AlpamayoR1Plugin()
