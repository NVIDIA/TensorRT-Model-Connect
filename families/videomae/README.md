# VideoMAE

This family builds the transformers VideoMAE pre-training model
(`VideoMAEForPreTraining`, `MCG-NJU/videomae-base` in the manifest) as one
float32 TensorRT engine for the `masked_video_text_to_video` Task: it
reconstructs the masked tubes of a clip from the visible ones. Batch is 1.
Precision is fp32, the checkpoint's own; other precisions and quantization are
refused.

The engine runs the model as the reference does: the tubelet embedding of the
whole clip, the encoder on the visible tubes only, then the decoder on the
visible tubes followed by one mask token per masked tube, and the pixel head
on the masked tubes.

## Request

- `source`: exactly the checkpoint's `num_frames` RGB frames of `image_size` x
  `image_size` pixels (16 frames of 224 x 224 for `videomae-base`), 8-bit or
  float32 in [0, 1]. The family does not resample: resize and center-crop the
  frames the way the checkpoint's processor does first. Timestamps are
  optional and are returned unchanged.
- `mask`: one value per source pixel, 1 where the model reconstructs and 0
  where the source conditions it. VideoMAE works on tubes of `tubelet_size`
  frames x `patch_size` x `patch_size` pixels (2 x 16 x 16 here), so each tube
  must be all 0 or all 1, and at least one tube of each kind is required. Soft
  mask values are refused.
- `prompt`: must be empty; the model has no text condition.
- There are no runtime Config fields.

## Result

One float32 RGB frame in [0, 1] per source frame, with the source timeline and
no context prefix. Conditioning tubes keep their source pixels exactly.

The checkpoint predicts each masked tube normalized by that tube's own
per-channel mean and unbiased standard deviation (`norm_pix_loss`). Like the
authors' visualization (`run_videomae_vis.py` in
[MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE)), the family restores
those statistics from the tube's source pixels (prediction x (std + 1e-6) +
mean) and clamps the result to [0, 1]. The source therefore has to hold the
clip's real pixels under the mask as well: the mask selects which tubes the
model re-creates, and the brightness and contrast of each re-created tube still
come from the source.

## Native API

The shared CLI has no command for this Task. Use the C or C++ SDK
(`trtmc::MaskedVideoTextToVideo`); `tests/sdk_consumer.c` and
`tests/sdk_consumer.cpp` are complete examples.

## Not supported

- `VideoMAEForVideoClassification` and `VideoMAEModel` checkpoints: they have no
  pixel decoder, and the shared SDK has no video classification or video
  feature Task. They are not claimed by this family.
- Checkpoints whose targets are not normalized per tube (`norm_pix_loss` false),
  or with another activation, without the q and v biases, or with other than
  three channels.
- Other clip lengths or frame sizes than the checkpoint's, batches, and
  precisions other than fp32.

## Checkpoint license and sources

`MCG-NJU/videomae-base` is published under CC BY-NC 4.0 (non-commercial use); the
repository does not include its weights. Sources of the network and of the
reconstruction method are listed in `NOTICE`.
