# CLIPSeg

The native image processor resizes the complete RGB image directly to the
checkpoint's square input. It preserves Pillow's integer coefficient and byte
rounding rules. Resize workspace uses one source row, four filtered rows,
axis coefficient tables and the fixed output; it does not allocate a
source-height by output-width intermediate image. Invalid image metadata and
preprocessing errors are rejected before either engine runs.

Applications accepting untrusted inputs should enforce prompt byte and image
pixel admission limits before calling the task. Tokenization and resizing still
take work proportional to the admitted input, and the encoded-token profile
limit is a separate compatibility check after tokenization.

Native Unicode tokenization requires ICU `uc` development libraries. Explicit
selection with `-DTRTMC_BUILD_FAMILIES=clipseg` fails at configuration time when
ICU is absent. Default builds may skip the optional family; explicitly building
`trtmc_model_clipseg` then reports the missing prerequisite.

Reference preprocessing requires Pillow 13.x. Native resizing follows its
shape-dependent pass order, including vertical-first filtering for strongly
anisotropic downsampling, with byte rounding between passes. The native tests
include independent Pillow 13 pixel goldens and a tall-image allocation bound.

Concurrent calls on one loaded pipeline serialize tokenization, both engines
and the final mask copy. Separately loaded pipelines keep independent locks.
