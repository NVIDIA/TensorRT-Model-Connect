# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned, non-learned coefficients for native reference preprocessing."""

import numpy as np


def coefficients():
    import torch
    from librosa.filters import mel
    from torchaudio.compliance.kaldi import get_mel_banks

    bank, _ = get_mel_banks(80, 512, 16000, 20, 0, 100, -500, 1)
    bank = torch.nn.functional.pad(bank, (0, 1))
    return dict(
        schema=1,
        kaldi_window=(torch.hann_window(400, periodic=False).pow(0.85)).tolist(),
        whisper_window=torch.hann_window(400).tolist(),
        acoustic_window=torch.hann_window(1920).tolist(),
        kaldi_bank=bank.flatten().tolist(),
        whisper_bank=mel(sr=16000, n_fft=400, n_mels=128).astype(np.float32).flatten().tolist(),
        acoustic_bank=mel(sr=24000, n_fft=1920, n_mels=80).astype(np.float32).flatten().tolist(),
    )
