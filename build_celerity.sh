#!/bin/bash

set -e  # Stop on first error

BASE_DIR="/home/malte.braig/testApproach/celerity-runtime"
mkdir -p "$BASE_DIR"

TIMESTAMP=$(date -u +%Y-%m-%d_%H-%M-%S)
BUILD_DIR="$BASE_DIR/build_$TIMESTAMP"
LOGFILE="$BASE_DIR/build_logs/build_$TIMESTAMP.log"

echo "📁 Creating isolated build directory: $BUILD_DIR" | tee -a "$LOGFILE"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "🔧 Running CMake..." | tee -a "$LOGFILE"
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCELERITY_SYCL_IMPL=DPC++ \
  -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx \
  2>&1 | tee -a "$LOGFILE"

echo "🚀 Building with Ninja..." | tee -a "$LOGFILE"
ninja 2>&1 | tee -a "$LOGFILE"

echo "✅ Build complete!" | tee -a "$LOGFILE"
