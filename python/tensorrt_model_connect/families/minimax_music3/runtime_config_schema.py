# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema for the ``music_minimax_music3`` namespace.

MiniMax-Music3 takes two texts: the lyrics to sing and a description of the
music to sing them over. The ``text_to_audio`` request carries exactly one
``prompt``, and the lyrics have to be the one that occupies it, because the
``tts_audio`` contract transcribes the generated audio and scores it against
``prompt``. Folding the description into the same string would make that score
meaningless.

The description therefore arrives through this namespace. Existing string
fields elsewhere in the repository hold short values -- a dump path, an
attention-mode enum -- so a description of a few thousand characters is a new
use of the channel rather than an established one. See the note on
:class:`~.plugin.MiniMaxMusic3Plugin` for the alternative worth proposing:
a second, optional text input on the shared ``text_to_audio`` request.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)

_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})

#: Upstream documents a 5,000-token text-prompt limit and a 9,000-frame audio
#: limit. The character bound here is a coarse guard against an obviously
#: wrong value, not a tokenizer-accurate check.
MAX_CAPTION_CHARS = 20000
MAX_AUDIO_FRAMES = 9000

SCHEMA = Schema(
    namespace="music_minimax_music3",
    fields=(
        ConfigField(
            name="caption",
            type_tag="string",
            default="",  # empty -> unconditioned on a description
            allowed_layers=_SESSION,
            validator=lambda value: (
                isinstance(value, str) and len(value) <= MAX_CAPTION_CHARS
            ),
        ),
        ConfigField(
            name="max_frames",
            type_tag="int32",
            default=9000,  # MAX_AUDIO_FRAMES, spelled out to match the C++ side
            allowed_layers=_SESSION,
            validator=lambda value: (
                isinstance(value, int) and 1 <= value <= MAX_AUDIO_FRAMES
            ),
        ),
        ConfigField(
            name="seed",
            type_tag="int64",
            default=-1,  # -1 -> use the default RNG state
            allowed_layers=_SESSION,
        ),
    ),
)

register_schema(SCHEMA)
