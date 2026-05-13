#!/usr/bin/env bash
set -euo pipefail

TRT11_ROOT="${TRT11_ROOT:-/opt/tensorrt-11}"
NCCL_ROOT="${NCCL_ROOT:-/opt/nccl-2.30}"

python_tag="$(python3 - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"

case "$(uname -m)" in
  x86_64)
    wheel_arch="linux_x86_64"
    ;;
  aarch64)
    wheel_arch="linux_aarch64"
    ;;
  *)
    echo "Unsupported container architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

wheels=(
  "${TRT11_ROOT}/python/tensorrt_dispatch-11.0.0.103-${python_tag}-none-${wheel_arch}.whl"
  "${TRT11_ROOT}/python/tensorrt_lean-11.0.0.103-${python_tag}-none-${wheel_arch}.whl"
  "${TRT11_ROOT}/python/tensorrt-11.0.0.103-${python_tag}-none-${wheel_arch}.whl"
)

for wheel in "${wheels[@]}"; do
  if [ ! -f "$wheel" ]; then
    echo "Missing TensorRT 11 Python wheel: $wheel" >&2
    exit 1
  fi
done

python3 -m pip install --disable-pip-version-check --force-reinstall --no-deps "${wheels[@]}"

export TRTMC_TRT_INCLUDE_DIR="${TRTMC_TRT_INCLUDE_DIR:-${TRT11_ROOT}/include}"
export TRTMC_TRT_LIBRARY="${TRTMC_TRT_LIBRARY:-${TRT11_ROOT}/lib/libnvinfer.so}"
export TRT_INC_DIR="${TRT_INC_DIR:-${TRTMC_TRT_INCLUDE_DIR}}"
export TRT_LIB_DIR="${TRT_LIB_DIR:-${TRT11_ROOT}/lib}"
export TRTMC_NCCL_LIB_DIR="${TRTMC_NCCL_LIB_DIR:-${NCCL_ROOT}/lib}"
export LD_LIBRARY_PATH="${TRTMC_NCCL_LIB_DIR}:${TRT_LIB_DIR}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/workspace/tensorrt-model-connect/tensorrt_model_connect:${PYTHONPATH:-}"

mkdir -p /tmp/trtmc-bin
printf '%s\n' '#!/usr/bin/env bash' 'exec python3 -m tensorrt_model_connect "$@"' > /tmp/trtmc-bin/trtmc-build
chmod +x /tmp/trtmc-bin/trtmc-build
export PATH="/tmp/trtmc-bin:${PATH}"

python3 - <<'PY'
import tensorrt as trt

assert trt.__version__ == "11.0.0.103", trt.__version__
assert hasattr(trt.INetworkDefinition, "add_dist_collective")
print(f"TensorRT Python ready: {trt.__version__}")
PY

if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec "$@"
