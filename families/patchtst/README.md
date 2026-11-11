# PatchTST

The checkpoint head selects the bundle's primary Task:

| Checkpoint head | Task | Result |
| --- | --- | --- |
| Prediction | `series_to_point_forecast` | `[horizon, channel]` values and positive horizon offsets |
| Regression with `loss="mse"` | `series_to_regression_values` | One value per target |
| Distribution regression | `series_to_regression_distribution` | Named parameter vectors, one value per target |

Each bundle exposes only its own Task. Normal parameters are `location` and
`scale`; Student-t parameters are `degrees_of_freedom`, `location`, and `scale`;
negative-binomial parameters are `total_count` and `logits`. Unknown channel or
target names and units remain unspecified. Classification is unsupported.

History is oldest first in `[time, input_channel]` order. A flat history infers
the channel count from the checkpoint. The runtime left-pads short histories
and truncates long histories by complete timesteps, applying the same operation
to the optional binary observed mask. Padding is unobserved. The optional I64
`frequency` configuration defaults to zero; nonzero values are rejected because
PatchTST has no frequency embedding.

FP16 and FP32 builds retain the existing graph and precision selectors. Results
export float32 values, including FP16 regression heads. Rebuild legacy
`time_series_forecast` bundles and deploy the family, runtime, and backend
together. The existing TP implementation selects replicated rank plans; this
migration does not introduce sharding or establish scaling performance.

The family tests retain the pinned Granite forecast and ETTh1 regression
checkpoints, all ten ETTh1 windows, and their existing numerical gates. They also
exercise masked histories, public C/C++ SDK consumers, and a fixed-seed official
MSE regression model in FP32 and FP16. The SDK consumers check Task discovery,
configuration errors, complete result metadata, and result ownership after
subsequent calls and model destruction. Build with `TRTMC_BUILD_TESTS=ON` and
provide `TRTMC_NATIVE_BUILD_DIR` when running selected E2E cases.
