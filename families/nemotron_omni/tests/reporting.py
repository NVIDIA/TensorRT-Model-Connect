# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display the exact input pixels and PCM already prepared for this family."""

from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_input_media(requests):
    if not evidence_enabled():
        return
    seen, media = set(), []
    try:
        import numpy as np
        from PIL import Image
        import wave

        for request in requests:
            images = ([request["image"]] if "image" in request else [])
            images += request.get("video", {}).get("frames", [])
            for spec in images:
                source = Path(spec["path"])
                if source in seen:
                    continue
                seen.add(source)
                pixels = np.fromfile(source, dtype=np.uint8).reshape(spec["height"], spec["width"], 3)
                path = source.with_suffix(".preview.png")
                Image.fromarray(pixels).save(path)
                media.append({"image": path, "source": source.name, "request": request.get("id", request.get("prompt"))})
            if "audio" in request:
                spec = request["audio"]
                source = Path(spec["path"])
                if source in seen:
                    continue
                seen.add(source)
                samples = np.fromfile(source, dtype="<f4")
                path = source.with_suffix(".preview.wav")
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(spec["channels"])
                    output.setsampwidth(2)
                    output.setframerate(spec["sample_rate"])
                    output.writeframes(np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes())
                media.append({"audio": path, "source": source.name, "preview": "PCM16 display of the original float input; inference samples are unchanged."})
        record_evidence("inputs", {"media": media})
    except Exception as error:
        record_evidence("inputs", {"media": media, "preview_unavailable": f"{type(error).__name__}: {error}"})
