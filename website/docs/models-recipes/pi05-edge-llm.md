---
title: Pi0.5 action chunks with Edge-LLM
---

The generic LeRobot builder and the openpi family command accept some of the
same weights, but produce different action contracts. Select the openpi policy
explicitly. For example, LIBERO generates ten 7-dimensional robot actions:

~~~sh
trtmc lerobot_pi05 build lerobot/pi05_libero_base -o libero.bundle \
  --pi05-policy-config pi05_libero --precision fp16 \
  --edge-builder-python /path/to/edge-env/bin/python
trtmc inspect libero.bundle
~~~

Edge execution requires Model Connect's matching **0.12.0 Pi0.5 C++ add-on**;
installing the Python wheel alone is not sufficient. In a
[Model Connect source build](../getting-started/source-build.md), select
`-DTRTMC_ENABLE_EDGELLM=ON -DTRTMC_EDGELLM_RELEASE=0.12.0 -DTRTMC_EDGELLM_POLICY_SDK=ON`
to provision or validate the matching native components. If the add-on is absent
or the eligible Edge build fails, the family warns and builds the same openpi
policy with native TensorRT. It does not switch to LeRobot preprocessing.
The bundle's sections distinguish Edge (`edge_llm.json`) from native
(`prefix.plan` and `denoise.plan`).

At inference, use the C++ `trtmc::ImagesStateTextToActionChunk` task with
named camera images, robot state and an instruction. The
[family recipe](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/families/lerobot_pi05/README.md#selecting-the-openpi-contract)
lists the policy-specific inputs and horizons, offline assets, and the
independent LIBERO reference. The recorded-frame comparison is a numerical
validation, not a closed-loop robot evaluation.

## Recorded action comparisons

Each row compares FP16 robot actions with independent FP32 reference actions
on **one fixed observation per policy**. Both Model Connect routes used the
same images, state, instruction, and initial noise for a given policy.

| Policy | Action shape | Native max absolute error | Edge max absolute error |
| --- | --- | ---: | ---: |
| `pi05_libero` | 10 × 7 | 0.000681 | 0.000849 |
| `pi05_droid` | 15 × 8 | 0.002371 | 0.002263 |
| `pi05_aloha` | 50 × 14 | 0.002264 | 0.001726 |

For DROID and ALOHA, the conversion from the pinned video frames to RGB
reproduced the original reference pixels exactly on both x86 and ARM.
The comparisons do not cover other episodes or closed-loop performance.
