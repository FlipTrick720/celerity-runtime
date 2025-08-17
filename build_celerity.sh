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
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCELERITY_SYCL_IMPL=DPC++ \
  -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/latest/ \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx \
  -DMPI_C_COMPILER=/usr/bin/mpicc \
  -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
  -DCELERITY_DETAIL_ENABLE_DEBUG=1 \
  -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-rdynamic"
  2>&1 | tee -a "$LOGFILE"

echo "🚀 Building with Ninja..." | tee -a "$LOGFILE"
ninja 2>&1 | tee -a "$LOGFILE"

echo "✅ Build complete!" | tee -a "$LOGFILE"

