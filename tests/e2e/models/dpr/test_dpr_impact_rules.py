"""DPR-owned impact narrowing rules."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import test_impact  # noqa: E402


def test_dpr_reference_rule_refines_hf_reference_diff() -> None:
    """DPR reference routing changes should select the DPR context encoder."""
    imap = test_impact.build_impact_map(REPO_ROOT)
    diff_text = """
diff --git a/tests/e2e_harness/references/hf_transformers.py b/tests/e2e_harness/references/hf_transformers.py
@@ -1 +1 @@
-            tokenizer = AutoTokenizer.from_pretrained(
-                model_ref, trust_remote_code=trust_remote_code)
+                # AutoTokenizer/AutoModel route this context checkpoint through
+                # the DPR question classes in transformers 5.x. Use the
+                # context fast tokenizer so HF sees the same token ids as the
+                # tokenizer.json bundled into the TRT artifact.
+                from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
+                tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
+                    model_ref, trust_remote_code=trust_remote_code)
+                model = _dpr.ctx_encoder.bert_model
+                tokenizer = AutoTokenizer.from_pretrained(
+                    model_ref, trust_remote_code=trust_remote_code)
"""

    broad = test_impact.classify_file(
        "tests/e2e_harness/references/hf_transformers.py",
        imap,
    )
    refined = test_impact.maybe_refine_match_with_diff(
        "tests/e2e_harness/references/hf_transformers.py",
        broad,
        diff_text,
        imap,
    )

    assert refined.rule == "dpr_reference_context_encoder"
    assert refined.models == ["dpr-ctx-encoder"]
