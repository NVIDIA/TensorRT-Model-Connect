"""Compatibility alias for :mod:`tensorrt_model_connect.build_cli`."""

from __future__ import annotations

import sys

from . import build_cli as _build_cli

sys.modules[__name__] = _build_cli
