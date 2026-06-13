"""LTX-2 audio VAE decoder TRT builder (stub).

Builds ``AutoencoderKLLTX2Audio`` decode path: mel-spectrogram-domain
audio latents -> mel-spectrogram. The downstream vocoder turns the
mel-spectrogram into a waveform.

Not yet implemented.
"""

from __future__ import annotations


def load_ltx2_audio_vae_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_audio_vae_weights: implement once the LTX-2 audio VAE "
        "checkpoint key map is known"
    )


def build_ltx2_audio_vae_decoder_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_audio_vae_decoder_engine: not implemented"
    )
