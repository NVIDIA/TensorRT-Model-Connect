# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import boltz
import numba
import numpy
import rdkit
import torch
import wandb
from boltz.model.models.boltz2 import Boltz2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Boltz-2 build profile check failed: {message}")


for package, expected in (
    ("boltz", "2.2.1"),
    ("numpy", "1.26.4"),
    ("pytorch-lightning", "2.5.0"),
    ("rdkit", "2026.3.5"),
    ("gemmi", "0.6.5"),
    ("numba", "0.61.0"),
    ("wandb", "0.18.7"),
    ("docker-pycreds", "0.4.0"),
    ("GitPython", "3.1.61"),
    ("gitdb", "4.0.12"),
    ("smmap", "5.0.3"),
    ("setproctitle", "1.3.7"),
    ("setuptools", "80.9.0"),
):
    installed = version(package)
    _require(installed == expected, f"{package}=={installed}, expected {expected}")

_require(numpy.__version__ == "1.26.4", f"numpy runtime {numpy.__version__}")
_require(rdkit.__version__ == "2026.03.5", f"rdkit runtime {rdkit.__version__}")
_require(numba.__version__ == "0.61.0", f"numba runtime {numba.__version__}")
_require(wandb.__version__ == "0.18.7", f"wandb runtime {wandb.__version__}")
_require(torch.__version__ == "2.12.0+cu130", f"torch runtime {torch.__version__}")
_require(Boltz2.__name__ == "Boltz2", "Boltz2 model class is unavailable")
_require(bool(boltz.__path__), "boltz package path is unavailable")
print(
    f"boltz={version('boltz')} torch={torch.__version__} "
    f"numpy={numpy.__version__} rdkit={rdkit.__version__}"
)
