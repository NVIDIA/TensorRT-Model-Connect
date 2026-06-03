"""LTX-2 vocoder TRT builder (stub).

Builds ``LTX2Vocoder`` (mel-spectrogram -> waveform). The exact
architecture (HiFi-GAN, BigVGAN, ...) is read from ``vocoder/config.json``;
plumb the matching graph_ops primitives into the TRT network once the
weight layout is confirmed on GPU.

Not yet implemented.
"""

from __future__ import annotations


def load_ltx2_vocoder_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_vocoder_weights: implement once the LTX-2 vocoder "
        "checkpoint key map is known"
    )


def build_ltx2_vocoder_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_vocoder_engine: not implemented"
    )
