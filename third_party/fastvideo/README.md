# FastVideo Windows H3 VSA patch

`windows-h3-vsa.patch` is a source patch for the public FastVideo repository:

- upstream: <https://github.com/hao-ai-lab/FastVideo>
- base revision: `3d8ac9d14bd697a89ede8f170cbfbca012a9edcc`
- license: Apache License 2.0; see `LICENSE` in this directory

The patch contains the Windows single-GPU process fixes, constructor-time
FastH3 adapter loading fixes, and the compile-safe SM121 tile-64 Triton VSA
route used by `scripts/install_windows_h3_fastvideo_vsa.ps1`. It also carries
focused upstream tests. The exact allowed path set and patch SHA-256 are pinned
in `tests/e2e/models/minimax_h3/fastvideo_windows_vsa_profile.json`.

The installer checks the patch hash, applies it only to the exact base
revision, and rejects any changed path outside that profile. This directory
does not contain FastVideo source trees, model weights, adapters, Python
environments, generated media, or compiled caches.
