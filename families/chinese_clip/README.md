# Chinese-CLIP

The family owns OFA-Sys Chinese-CLIP image/text embedding, including the BERT
text tower, vision transformer, WordPiece tokenizer, and Pillow-compatible
bicubic image preprocessing. The pinned checkpoint's processor resizes directly
to its configured square dimensions without a center crop.

Native preprocessing uses address-sized buffer offsets and clamps resampling
support before converting coordinates to integer indices. Callers must provide
the full source image storage described by the SDK request.

Applications accepting untrusted images should enforce a source-pixel and
working-memory budget before invoking the SDK. Resizing first constructs a
three-channel byte image of the source dimensions, then a horizontal byte buffer
with `source_height * output_width * 3` elements. The fixed model output size
does not bound these allocations. Allocation failure is reported as a request
failure; source-image admission limits belong to the hosting application.
