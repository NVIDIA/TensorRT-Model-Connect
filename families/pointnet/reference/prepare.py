# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the PointNet S3DIS model directory under ``<environment>/trtmc-reference/pointnet-s3dis``: the
upstream checkpoint (the catalog's ``best_model.pth`` at its pinned upstream commit), the ONNX export the builder reads
and the S3DIS shape contract, as tests/test_e2e.py prepares them. The bundle builds from this directory and the
native reference loads the same checkpoint."""

from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY.parents[1]))

from families.pointnet.tests.official_reference import export_onnx  # noqa: E402

CHECKPOINT = ("https://raw.githubusercontent.com/yanx27/Pointnet_Pointnet2_pytorch/"
              "eb64fe0b4c24055559cea26299cb485dcb43d8dd/log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth")


def main() -> None:
    destination = Path(sys.prefix) / "trtmc-reference" / "pointnet-s3dis"
    if (destination / "pointnet.onnx").is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pointnet.", dir=destination.parent))
    try:
        checkpoint = temporary / "best_model.pth"
        with urllib.request.urlopen(CHECKPOINT, timeout=600) as response, open(checkpoint, "wb") as handle:
            shutil.copyfileobj(response, handle)
        export_onnx(temporary, temporary / "pointnet.onnx")
        shutil.copyfile(FAMILY / "tests/data/config.json", temporary / "config.json")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
