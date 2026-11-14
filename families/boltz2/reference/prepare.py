# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Boltz-2 package root (model_config.resolve_package_root) the bundle builds from and the native
reference reads, under ``<environment>/trtmc-reference/boltz2-model``, as the E2E test stages it: the pinned public
checkpoints and molecule archive (linked from the Hugging Face cache, the archive extracted), the packaged monomer
request with its MSA, and that request preprocessed by the pinned ``boltz``."""

from pathlib import Path
import json
import os
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from families.boltz2.model_config import MOLS, MOLS_ARCHIVE, resolve_package_root  # noqa: E402

DESTINATION = Path(sys.prefix) / "trtmc-reference" / "boltz2-model"
FILES = ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar")


def main() -> int:
    if resolve_package_root(DESTINATION) is not None:
        return 0
    from boltz.main import process_inputs
    from huggingface_hub import snapshot_download

    manifest = json.loads((ROOT / "families/boltz2/tests/manifests/boltz2-protein-monomer.json").read_text())
    snapshot = Path(snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"], allow_patterns=list(FILES)))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="boltz2-model.", dir=DESTINATION.parent))
    try:
        for name in FILES:
            (work / name).symlink_to((snapshot / name).resolve(strict=True))
        for name in ("protein_monomer.yaml", "protein_monomer.a3m"):
            shutil.copy2(ROOT / "families/boltz2/tests/data" / name, work)
        with tarfile.open(work / MOLS_ARCHIVE, "r") as archive:
            archive.extractall(work, filter="data")
        previous = Path.cwd()
        os.chdir(work)
        try:
            process_inputs(data=[work / "protein_monomer.yaml"], out_dir=work, ccd_path=work / MOLS / "unused.pkl",
                           mol_dir=work / MOLS, msa_server_url="https://api.colabfold.com",
                           msa_pairing_strategy="greedy", max_msa_seqs=1024, use_msa_server=False, boltz2=True,
                           preprocessing_threads=1)
        finally:
            os.chdir(previous)
        if resolve_package_root(work) is None:
            raise RuntimeError("the staged Boltz-2 package is incomplete")
        work.rename(DESTINATION)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
