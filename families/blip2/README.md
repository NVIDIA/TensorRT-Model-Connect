# BLIP-2

The native image processor resizes the complete RGB image directly to the
checkpoint's square input. It preserves Pillow's bicubic coefficient and byte
rounding rules. Resize workspace uses one source row, four filtered rows,
axis coefficient tables and the fixed output; it does not allocate a
source-height by output-width intermediate image. This keeps tall, narrow
images from multiplying worker memory by the output width.

Applications accepting untrusted images should enforce source pixel and prompt
byte admission limits before calling the task. Input processing still takes
work proportional to the admitted input; the fixed inference profile and
generation limit do not replace application admission controls.

Reference preprocessing requires Pillow 13.x. Native resizing follows its
shape-dependent pass order, including vertical-first filtering for strongly
anisotropic downsampling, with byte rounding between passes. The native tests
include independent Pillow 13 pixel goldens and a tall-image allocation bound.
