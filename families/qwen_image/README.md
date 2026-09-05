# Qwen-Image build geometry

`image_generation` uses `BuildRequest.image_height` and `image_width` as the
static output size. For the currently supported `image_edit` build, those two
required fields are the raw condition-image size; the output stays at the
checkpoint's default 1024 x 1024 size. The direct family E2E reads the real
condition fixture dimensions before calling `build()`.
