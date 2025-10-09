#!/bin/bash
# Quick build and test script - runs everything in one go

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Celerity Quick Build & Test"
echo "=========================================="
echo

# Check if we need to build
LATEST_BUILD=$(ls -d build_* 2>/dev/null | sort | tail -n1 || true)
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
echo "🧪 Running tests..."
./run_test.sh --profile test "$@"
