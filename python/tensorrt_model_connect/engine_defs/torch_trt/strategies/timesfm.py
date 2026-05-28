"""TimesFM build strategy for Torch-TRT.

This strategy keeps the I/O surface numeric instead of token-based:
  - past_values: dense float tensor [1, context_len]
  - past_values_padding: dense int tensor [1, context_len], where 1 means pad
  - freq: dense int tensor [1]

The wrapper avoids the high-level `TimesFmModelForPrediction.forward()` helper
because that path builds Python lists and new tensors from traced values, which
`torch.export` cannot capture reliably. Instead we call the decoder directly and
reuse the official post-processing logic on tensor-only inputs.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised indirectly when torch is installed
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - allows import on CPU-only test hosts
    torch = None

    class _FallbackModule:
        def __init__(self, *args, **kwargs):
            pass

    class _FallbackNN:
        Module = _FallbackModule

    nn = _FallbackNN()


class TimesFmWrapper(nn.Module):
    """Wrap TimesFmModelForPrediction with TRT-friendly tensor I/O.

    The wrapper intentionally keeps a fixed batch size of 1 for export/runtime
    simplicity. The runtime pipeline flattens a single input series into the
    same dense representation.
    """

    def __init__(
        self,
        model: nn.Module,
        context_length: int,
        *,
        compute_dtype=None,
    ):
        super().__init__()
        self.model = model
        self.context_length = context_length
        if torch is not None and compute_dtype is None:
            compute_dtype = torch.float16
        self.compute_dtype = compute_dtype

    def forward(self, past_values, past_values_padding, freq):
        if torch is None:
            raise ImportError("torch is required to execute TimesFmWrapper")
        values = past_values.to(self.compute_dtype)
        padding = past_values_padding.to(values.dtype)
        freq = freq.reshape(-1, 1).to(torch.long)

        context_len = min(values.shape[1], self.context_length)
        final_out = values[:, -context_len:]
        current_padding = padding[:, -context_len:]

        decoder_output = self.model.decoder(
            past_values=final_out,
            past_values_padding=current_padding,
            freq=freq,
            output_attentions=False,
            output_hidden_states=False,
        )
        fprop_outputs = self.model._postprocess_output(
            decoder_output.last_hidden_state,
            (decoder_output.loc, decoder_output.scale),
        )

        output_patch_len = int(self.model.config.horizon_length)
        full_outputs = fprop_outputs[:, -1, :output_patch_len, :]
        full_outputs = full_outputs[:, : self.model.config.horizon_length, :]
        mean_outputs = full_outputs[:, :, 0]

        return (
            mean_outputs.to(torch.float32),
            full_outputs.to(torch.float32),
        )


class TimesFmBuildStrategy:
    """Build strategy for TimesFM."""

    name = "timesfm"
    runtime_strategy = "timesfm_torchtrt"

    def wrap_model(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        if compute_dtype is None:
            compute_dtype = torch.float16

        context_length = getattr(config, "context_length", 0) or max_cache_length
        if context_length <= 0:
            context_length = max_cache_length
        return TimesFmWrapper(model, int(context_length), compute_dtype=compute_dtype)

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        if torch is None:
            raise ImportError("torch is required to build TimesFM export args")
        del precision

        context_length = getattr(config, "context_length", 0) or max_cache_length
        if context_length <= 0:
            context_length = max_cache_length

        device = "cuda" if torch.cuda.is_available() else "cpu"
        valid_len = min(int(context_length), 12)
        past_values = torch.zeros((1, int(context_length)), dtype=torch.float32, device=device)
        if valid_len > 0:
            past_values[:, -valid_len:] = torch.linspace(
                0.0,
                1.0,
                steps=valid_len,
                dtype=torch.float32,
                device=device,
            )
        past_values_padding = torch.ones(
            (1, int(context_length)), dtype=torch.int32, device=device)
        if valid_len > 0:
            past_values_padding[:, -valid_len:] = 0
        freq = torch.full((1,), 2, dtype=torch.int32, device=device)
        return (past_values, past_values_padding, freq)

    def pre_export_setup(self) -> None:
        pass
