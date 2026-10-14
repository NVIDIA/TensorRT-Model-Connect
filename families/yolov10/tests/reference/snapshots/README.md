# Reference measurements

These numeric predictions were recorded with Ultralytics 8.3.0, PyTorch
2.12.0, and NumPy 1.26.4. They use the checkpoint loading and letterbox inputs
of the E2E worker at commit `647ab30a5ec1c1b623e31972d28c0f8a6c7a8b55`.
No upstream implementation or weights are included. Each JSON record identifies
the checkpoint files and test image by SHA-256 and records the pinned Hugging
Face revision. Hashes are provenance receipts; runtime selection uses the
existing revision-pinned test manifests.

`original` uses `tests/data/test_img.jpeg`; `mirrored` flips that RGB image
horizontally before the same 640-pixel letterbox preprocessing. After the
upstream model returns `(ranked, auxiliary)`, capture the decoded cell tensor
with `model.model[-1]._inference(auxiliary["one2one"])`. This tensor has shape
`[1, 84, 8400]`: four box coordinates and 80 class probabilities, in fixed
spatial order, before either top-k selection.

Each case retains the final ranked-output shape and accepted detections at
confidence 0.25. It also records the decoded cell shape and 256 evenly spaced
samples of its flattened values. Comparing cells before ranking avoids treating
swapped, nearly tied low-confidence predictions as coordinate errors.

The E2E worker checks both cell predictions and accepted detections with
`atol=1e-4` and `rtol=1e-5`. Native count, class, score and coordinate thresholds
are unchanged. Do not regenerate these records from the local reference or the
TensorRT builder: updates require an independently validated oracle and matching
checkpoint/image provenance.
