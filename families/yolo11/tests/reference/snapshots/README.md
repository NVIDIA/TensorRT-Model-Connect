# Reference measurements

These numeric predictions were recorded with Ultralytics 8.3.0 and the E2E
reference workers at commit `647ab30a5ec1c1b623e31972d28c0f8a6c7a8b55` before
removing that package from the tests. No upstream implementation or weights
are included. Each JSON record identifies the checkpoint files and test image
by SHA-256 and records the pinned Hugging Face revision and PyTorch version
used for capture. Hashes are provenance receipts; runtime selection continues
to use the existing revision-pinned test manifests.

`original` uses `tests/data/test_img.jpeg`; `mirrored` flips that RGB image
horizontally before the same 640-pixel letterbox preprocessing. Each case
contains the raw prediction shape, 256 evenly spaced samples of its flattened
values, and all detections at confidence 0.25. Raw values are captured before
suppression, which can modify coordinates in place in the old implementation.

The E2E worker checks the local PyTorch reference against these records, using
`atol=1e-4` and `rtol=1e-5`, before comparing it with the native runtime. Native
count, class, score and coordinate thresholds are unchanged. Do not regenerate
these records from the local reference or the TensorRT builder: updates require
an independently validated oracle and matching checkpoint/image provenance.
