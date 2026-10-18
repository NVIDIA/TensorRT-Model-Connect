# PersonaPlex audio runtime

Input resampling follows the qualification Native loader: librosa's `soxr_hq`
recipe, with its floating-point output-length rule and zero padding. An
approximate filter can change Mimi input codes even when waveform differences
are small.

Non-24 kHz input requires the system `libsoxr.so.0` runtime (`apt-get install
libsoxr0` on Ubuntu). The family loads it when resampling is needed and reports
an error if it is unavailable. Already-24 kHz input bypasses resampling. The
family CI dependency image installs this runtime; it needs no development
headers or link-time dependency, and other model families do not load it.

The wheel does not bundle libsoxr. System libsoxr is a separate LGPL-2.1-or-later
dependency; see its [source and license](https://github.com/chirlu/soxr).

`tests/test_resampler_precision.py` checks the Native HQ waveform on a synthetic
signal and librosa's length/padding behavior. Its waveform checks require the
runtime library; identity and invalid-rate checks need only a C++ compiler.
