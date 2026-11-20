# OCR fixture

`invoice.png` is a project-created 672 × 448 RGB invoice containing invented
items, prices, and an invoice number. It is distributed under Apache-2.0 with
the project. No real customer or third-party document is included.

`generate_fixture.py` renders the image using Pillow and DejaVu Sans. The
committed image is the test input; inference tests do not regenerate it.

The selected OCR case requires nonempty complete generation containing
`LIGHTON OCR`, `1042`, and `30.00`, with token IDs and decoded text identical
to the independent pinned official greedy reference. Image features and two
teacher-forced logit observations also have separately declared numerical
gates in the family thresholds file.
