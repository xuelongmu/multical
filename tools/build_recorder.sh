#!/usr/bin/env bash
set -euo pipefail
RECORDER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RECORDER_CUDA="${CUDA_PATH:-/usr/local/cuda}"
RECORDER_SPIN="${SPINNAKER_PATH:-/opt/spinnaker}"
mkdir -p "$RECORDER_ROOT/.native"
RECORDER_BUILD="$(mktemp "$RECORDER_ROOT/.native/.recorder-build.XXXXXX")"
trap 'rm -f "$RECORDER_BUILD"' EXIT
# .native/deps can hold extracted Ubuntu development packages, avoiding changes
# to the host installation. System-installed development headers work as well.
g++ -O3 -std=c++17 -pthread -Wall -Wextra \
  -I"$RECORDER_ROOT/.native/deps/usr/include" \
  -I"$RECORDER_ROOT/.native/deps/usr/include/x86_64-linux-gnu" \
  -isystem "$RECORDER_SPIN/include" -isystem "$RECORDER_CUDA/include" \
  "$RECORDER_ROOT/native/recorder/main.cpp" \
  -L"$RECORDER_SPIN/lib" -Wl,-rpath,"$RECORDER_SPIN/lib" \
  -L"$RECORDER_CUDA/lib64" -Wl,-rpath,"$RECORDER_CUDA/lib64" \
  -lSpinnaker -lnvjpeg -lnppicc -lnppc -lcudart \
  -l:libavformat.so.58 -l:libavcodec.so.58 -l:libavutil.so.56 \
  -o "$RECORDER_BUILD"
mv "$RECORDER_BUILD" "$RECORDER_ROOT/.native/multical-recorder"
echo "Built $RECORDER_ROOT/.native/multical-recorder"
