# Qwen3.5 image fixture

The car image is a family-owned copy of the repository's existing
Qwen-VL image fixture. It exercises image-conditioned color recognition in the
provisioned Qwen3.5 2B Edge TP1 and 27B Edge TP2 cases.

These cases require the optional Edge 0.11 SDK, its native family DSO, and one
or two compatible runtime GPUs respectively. They are explicitly selectable rather
than part of the generic premerge lane, which does not provision Edge.

Run from the repository root with TRTMC_BINARY and TRTMC_RUNTIME_ROOT pointing
to a matching native build, CMAKE_PREFIX_PATH pointing to the Edge SDK, and
TRTMC_E2E_ARTIFACT_DIR pointing to the evidence output directory:

    python -m pytest -q families/qwen3_5/tests/test_e2e.py \
      --e2e-testcase qwen35-2b-edge-image-tp1 \
      --e2e-testcase qwen35-27b-edge-image-tp2

Both cases pin the source checkpoint, require an Edge bundle (no fallback),
compare against FP32 Transformers with the existing normalized-edit-distance
gate of 0.15, and require the expected image-dependent answer.

The 27B FP32 oracle requires four visible GPUs with free placement budgets of
34, 34, 28, and 16 GiB, respectively. The first two are the compatible TP2
runtime pair; the other two provide reference capacity. CPU offload is disabled
because an offloaded reference diverged after the first token. Reference
precision and the quality gate are unchanged.

Use a fast temporary filesystem with at least 60 GiB free for the 27B runtime
bundle extraction (TMPDIR). Slow storage can consume the unchanged 600-second
runtime timeout before inference starts. Keep checkpoint and evidence storage
separate from the temporary extraction directory. Timeout diagnostics are retained.
