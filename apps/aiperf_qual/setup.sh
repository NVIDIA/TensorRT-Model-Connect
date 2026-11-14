#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prepare a machine (Docker or bare metal) for trtmc-aiperf-qual:
#
#   apps/aiperf_qual/setup.sh AIPERF_VENV [SERVE_PYTHON]
#
# 1. Creates the orchestrator environment AIPERF_VENV: AIPerf 0.13.0, the TRTMC AIPerf plugins, and
#    the metric import hook (PYTHON selects the interpreter, 3.11 or newer; default python3).
# 2. With SERVE_PYTHON (the interpreter that has torch, transformers, and the TRTMC runtime), installs
#    the trtmc-perf-serve packages into it.
#
# Then copy config/environments/example.yaml to config/environments/<machine>.yaml, fill in its paths,
# and check it with: trtmc-aiperf-qual doctor --environment config/environments/<machine>.yaml
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
venv=${1:?usage: setup.sh AIPERF_VENV [SERVE_PYTHON]}
serve=${2:-}

"${PYTHON:-python3}" -m venv "$venv"
"$venv/bin/python" -m pip install --quiet --upgrade pip
"$venv/bin/python" -m pip install --quiet -r "$here/requirements.txt"
"$venv/bin/python" -m pip install --quiet --no-deps "$here/plugins"
PYTHONPATH="$here" "$venv/bin/python" -m trtmc_aiperf_qual doctor --fix

if [ -n "$serve" ]; then
  "$serve" -m pip install --quiet -r "$repo/apps/perf_serving/requirements.txt"
  "$serve" -c "import torch, transformers; print('serve_python: torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
fi
echo "aiperf: $venv/bin/aiperf"
