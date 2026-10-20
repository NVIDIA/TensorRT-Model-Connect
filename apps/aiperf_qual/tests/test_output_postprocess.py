# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest

from trtmc_aiperf_qual.absolute import postprocess
from trtmc_aiperf_qual.config import Environment


def test_postprocess_adds_text_without_changing_timing_or_original_artifacts(tmp_path):
    script = tmp_path / "transcribe.py"
    script.write_text("import json,sys\nfrom pathlib import Path\n"
                      "d=json.loads(Path(sys.argv[2]).read_text())\n"
                      "for row in d.values(): row['text']='Option C'\n"
                      "Path(sys.argv[4]).write_text(json.dumps(d))\n")
    observations = {0: {"artifact": "unchanged.npy", "sample_rate": 24000}}
    item = {"postprocess": {"script": script.name}}
    environment = Environment({"repo": str(tmp_path), "serve_python": "/unused-serving-interpreter"})
    postprocess(environment, item, observations, tmp_path / "run", python=sys.executable)
    assert observations == {0: {"artifact": "unchanged.npy", "sample_rate": 24000, "text": "Option C"}}
    assert json.loads((tmp_path / "run.postprocess-input.json").read_text()) == {
        "0": {"artifact": "unchanged.npy", "sample_rate": 24000}}


@pytest.mark.parametrize("mutation", ["d.clear()", "d['0']['sample_rate']=16000"])
def test_postprocess_rejects_lost_samples_or_modified_evidence(tmp_path, mutation):
    script = tmp_path / "bad.py"
    script.write_text("import json,sys\nfrom pathlib import Path\n"
                      "d=json.loads(Path(sys.argv[2]).read_text())\n" + mutation + "\n"
                      "Path(sys.argv[4]).write_text(json.dumps(d))\n")
    environment = Environment({"repo": str(tmp_path), "serve_python": sys.executable})
    with pytest.raises(ValueError, match="postprocessing changed"):
        postprocess(environment, {"postprocess": {"script": script.name}},
                    {0: {"sample_rate": 24000}}, tmp_path / "run")
