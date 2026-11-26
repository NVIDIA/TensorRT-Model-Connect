# MiniCPM-OCR test images

Lossless PNG inputs of the family E2E, all derived from the project-created OCR screenshot
`families/deepseek_ocr/tests/data/orc_test_img.jpeg` (Apache License 2.0, see `ASSET_LICENSES.md`):

| File | Source region (columns x rows) | Use |
|---|---|---|
| `minicpm_ocr_page.png` | whole image, 931 x 468 | sliced input (overview plus a 1 x 2 slice grid) |
| `minicpm_ocr_line.png` | 0-869 x 0-35 | one long text line (unsliced, strongly elongated) |
| `minicpm_ocr_paragraph.png` | 0-930 x 52-199 | paragraph, first image of the two-image request |
| `minicpm_ocr_block.png` | 0-930 x 228-467 | indented list, second image of the two-image request (1 x 3 slices) |

The JPEG is decoded once with Pillow and stored as PNG so that the native loader and the reference processor read
the same pixels. The root `.gitignore` excludes `*.png`; these files are added explicitly.
