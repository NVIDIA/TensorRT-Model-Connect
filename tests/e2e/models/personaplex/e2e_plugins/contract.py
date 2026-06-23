"""PersonaPlex-owned speech-to-speech contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import MetricResult
from tests.e2e_harness.plugins.base import make_fail, make_pass


class PersonaPlexSpeechToSpeechPlugin:
    reference_families = ["s2s_personaplex"]
    user_contract = "speech_response"

    def configure_reference(self, case):
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_wav = trt_output.data.get("wav_path")
        trt_rms = trt_output.data.get("rms")
        trt_duration = trt_output.data.get("duration_s")
        min_rms = threshold.metrics.get("contract_min_rms", 0.001)

        metrics = {}
        has_wav = trt_wav is not None and isinstance(trt_wav, str) and len(trt_wav) > 0
        metrics["has_audio"] = MetricResult(
            value=1.0 if has_wav else 0.0,
            threshold=1.0,
            operator="==",
            passed=has_wav,
        )

        if trt_rms is not None:
            rms_ok = float(trt_rms) >= min_rms
            metrics["rms"] = MetricResult(
                value=float(trt_rms),
                threshold=min_rms,
                operator=">=",
                passed=rms_ok,
            )

        if trt_duration is not None:
            dur_ok = float(trt_duration) >= 0.1
            metrics["duration_s"] = MetricResult(
                value=float(trt_duration),
                threshold=0.1,
                operator=">=",
                passed=dur_ok,
            )

        trt_tokens = trt_output.data.get("token_ids")
        ref_tokens = ref_output.data.get("token_ids")
        if trt_tokens is not None and ref_tokens is not None:
            min_len = min(len(trt_tokens), len(ref_tokens))
            if min_len > 0:
                match = sum(
                    1
                    for actual, expected in zip(trt_tokens[:min_len], ref_tokens[:min_len])
                    if actual == expected
                ) / min_len
                token_threshold = threshold.metrics.get("contract_token_match", 0.5)
                metrics["token_match"] = MetricResult(
                    value=match,
                    threshold=token_threshold,
                    operator=">=",
                    passed=match >= token_threshold,
                )

        rule = "audio health + token match"
        if all(metric.passed for metric in metrics.values()):
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "PersonaPlex speech response health check failed",
        )


plugin = PersonaPlexSpeechToSpeechPlugin()
