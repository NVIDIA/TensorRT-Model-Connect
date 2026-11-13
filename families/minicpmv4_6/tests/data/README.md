# MiniCPM-V 4.6 image fixtures

- `test_img.jpeg` is a byte-identical copy of the repository's shared vehicle
  test photograph (see `ASSET_LICENSES.md`); it exercises a sliced image (a
  source thumbnail plus a 1 x 2 slice grid) and image-conditioned color
  recognition.
- `orc_test_img.jpeg` is a byte-identical copy of the project-created OCR
  screenshot (see `ASSET_LICENSES.md`); it exercises a wider slice grid and
  text reading.

The E2E decodes each fixture once with Pillow and writes it as lossless PNG, so
the native CLI, the SDK consumers and the Transformers reference read identical
RGB values regardless of their JPEG decoders.
