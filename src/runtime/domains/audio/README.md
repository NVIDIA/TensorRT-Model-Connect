# Audio Domain

Shared audio-domain helper implementations are retired. Generic public WAV I/O
lives in `include/trtmc/trtmc_io.hpp`; model-specific feature extraction,
post-processing, and audio runtime behavior live under each owning model family
in `src/runtime/models/<family>/`.
