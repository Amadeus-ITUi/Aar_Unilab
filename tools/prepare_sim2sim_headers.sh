#!/usr/bin/env bash
# Public SDK headers live in the SSD dependency cache, never in references/.
set -euo pipefail
sdk_cache=/ssd/conda/cache/aar_unilab-native/sim2sim-sdk
mkdir -p "$sdk_cache/onnxruntime-1.22.0" "$sdk_cache/glfw-3.4/GLFW"
for sdk_header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h; do
  sdk_target="$sdk_cache/onnxruntime-1.22.0/$sdk_header"
  if [[ ! -s "$sdk_target" ]]; then
    curl --fail --location --retry 3 "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/include/onnxruntime/core/session/$sdk_header" --output "$sdk_target.tmp"
    mv "$sdk_target.tmp" "$sdk_target"
  fi
done
for sdk_header in glfw3.h glfw3native.h; do
  sdk_target="$sdk_cache/glfw-3.4/GLFW/$sdk_header"
  if [[ ! -s "$sdk_target" ]]; then
    curl --fail --location --retry 3 "https://raw.githubusercontent.com/glfw/glfw/3.4/include/GLFW/$sdk_header" --output "$sdk_target.tmp"
    mv "$sdk_target.tmp" "$sdk_target"
  fi
done
