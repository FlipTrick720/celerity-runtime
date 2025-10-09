#!/bin/bash
# Quick build and test script - runs everything in one go

set -e

SCRIPT_DIR="$PWD"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Celerity Quick Build & Test"
echo "=========================================="
echo

# Check if we need to build
LATEST_BUILD=$(ls -d build_20* 2>/dev/null | sort | tail -n1 || true)
if [[ -z "$LATEST_BUILD" ]]; then
  echo "📦 No build found, building now..."
  ./build_celerity.sh
else
  echo "📦 Found existing build: $LATEST_BUILD"
  read -p "   Rebuild? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./build_celerity.sh
  fi
fi

echo
echo "🧪 Running tests via sbatch ... look in z_output.log for Logs"
sbatch run_test.sh --profile test
