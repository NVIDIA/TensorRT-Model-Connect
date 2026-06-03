"""LTX-2 text connectors TRT builder (stub).

Builds engines for ``LTX2TextConnectors`` — the small transformer stack
that projects the raw T5-v1_1-XXL hidden states (d_model=4096) into the
DiT caption inputs:

- ``video_connector``: produces caption_channels = 3840 hidden states
  for the 14B video DiT.
- ``audio_connector``: produces a separate hidden state stream for the
  5B audio DiT.

Layer counts and hidden sizes come from ``connectors/config.json``.

Not yet implemented. See ``plugin.LTX2Plugin.build_components`` for the
scaffolding plan.
"""

from __future__ import annotations


def load_ltx2_text_connectors_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_text_connectors_weights: implement once the LTX-2 "
        "connectors checkpoint key map is known"
    )


def build_ltx2_text_connectors_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_text_connectors_engine: implement once the LTX-2 "
        "C++ runtime exposes the connector hooks"
    )
