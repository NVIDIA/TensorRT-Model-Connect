# Chinese-CLIP

The family owns OFA-Sys Chinese-CLIP image/text embedding, including the BERT
text tower, vision transformer, WordPiece tokenizer, and Pillow-compatible
bicubic image preprocessing. The pinned checkpoint's processor resizes directly
to its configured square dimensions without a center crop.

Native preprocessing uses address-sized buffer offsets and clamps resampling
support before converting coordinates to integer indices. Callers must provide
the full source image storage described by the SDK request.

Resize workspace uses one source row, four filtered rows, axis coefficient
tables and the fixed output. It no longer allocates a full source byte image or
a `source_height * output_width * 3` intermediate. Horizontal byte rounding and
bicubic coefficients are unchanged.

Applications accepting untrusted inputs should enforce raw text byte and source
image pixel admission limits before calling either task. Tokenization and
resizing still take work proportional to the admitted input. The encoded-token
profile limit is checked after tokenization and is not a raw input budget.
