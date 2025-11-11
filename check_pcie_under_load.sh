#!/bin/bash
# Check PCIe configuration while GPU is under load
# This verifies if PCIe link speed/width changes during active transfers

set -e

echo "========================================================================"
echo "PCIe Configuration Check - Under Load Test"
echo "========================================================================"
echo ""

# Check if SYCL library is available
if ! ldconfig -p | grep -q libsycl.so; then
    echo "WARNING: libsycl.so not found in library path"
    echo "Attempting to source Intel oneAPI environment..."
    
    # Common oneAPI locations
    ONEAPI_PATHS=(
        "/opt/intel/oneapi/setvars.sh"
        "$HOME/intel/oneapi/setvars.sh"
        "/opt/intel/oneapi/compiler/latest/env/vars.sh"
    )
    
    SOURCED=false
    for SETVARS in "${ONEAPI_PATHS[@]}"; do
        if [ -f "$SETVARS" ]; then
            echo "Found: $SETVARS"
            source "$SETVARS" --force > /dev/null 2>&1 || true
            SOURCED=true
            break
        fi
    done
    
    if [ "$SOURCED" = false ]; then
        echo ""
        echo "ERROR: Could not find Intel oneAPI environment"
        echo "Please run: source /opt/intel/oneapi/setvars.sh"
        echo "Then run this script again"
        exit 1
    fi
    echo "✓ oneAPI environment loaded"
fi
echo ""

# Find Intel Arc GPU
GPU_PCI=$(lspci | grep -i "VGA.*Intel" | grep -i "Arc" | head -1 | cut -d' ' -f1)
if [ -z "$GPU_PCI" ]; then
    echo "ERROR: Intel Arc GPU not found!"
    exit 1
fi

echo "Found GPU at: $GPU_PCI"
SYSFS_PATH="/sys/bus/pci/devices/0000:$GPU_PCI"
echo "sysfs path: $SYSFS_PATH"
echo ""

# Function to read PCIe status
read_pcie_status() {
    SPEED=$(cat $SYSFS_PATH/current_link_speed 2>/dev/null || echo "N/A")
    WIDTH=$(cat $SYSFS_PATH/current_link_width 2>/dev/null || echo "N/A")
    echo "$SPEED x$WIDTH"
}

# Check idle state
echo "1. PCIe status when IDLE:"
IDLE_STATUS=$(read_pcie_status)
echo "   $IDLE_STATUS"
echo ""

# Find the benchmark executable
MEMCPY_BENCH="bench/build/memcpy_linear"
if [ ! -f "$MEMCPY_BENCH" ]; then
    echo "ERROR: memcpy_linear benchmark not found at $MEMCPY_BENCH"
    echo "Please build the benchmarks first with: cd bench && ./build_bench.sh"
    exit 1
fi

echo "2. Running benchmarks to stress PCIe link..."
echo ""

# Test 1: Batch transfers (sustained load)
echo "   Test 1: Batch memory transfers (64MB, 10 seconds)"
echo "   Running: $MEMCPY_BENCH --batch --min 67108864 --max 67108864 --steps 1 --secs 10"
$MEMCPY_BENCH --batch --min 67108864 --max 67108864 --steps 1 --secs 10 > /tmp/pcie_batch.log 2>&1 &
BENCH_PID=$!

sleep 2
echo ""
echo "3. Monitoring PCIe status during batch transfers..."
for i in {1..8}; do
    STATUS=$(read_pcie_status)
    echo "   [$i] $STATUS"
    sleep 1
done

wait $BENCH_PID 2>/dev/null || true
echo "   ✓ Batch test complete"
echo ""

# Test 2: Sweep different sizes (varied load)
echo "   Test 2: Size sweep (1KB to 64MB)"
echo "   Running: $MEMCPY_BENCH --min 1024 --max 67108864 --steps 16 --secs 1"
$MEMCPY_BENCH --min 1024 --max 67108864 --steps 16 --secs 1 > /tmp/pcie_sweep.log 2>&1 &
BENCH_PID=$!

sleep 2
echo ""
echo "4. Monitoring PCIe status during size sweep..."
for i in {1..8}; do
    STATUS=$(read_pcie_status)
    echo "   [$i] $STATUS"
    sleep 1
done

wait $BENCH_PID 2>/dev/null || true
echo "   ✓ Sweep test complete"
echo ""

# Test 3: Event overhead (kernel launch stress)
EVENT_BENCH="bench/build/event_overhead"
if [ -f "$EVENT_BENCH" ]; then
    echo "   Test 3: Event overhead (kernel launches)"
    echo "   Running: $EVENT_BENCH --secs 5"
    $EVENT_BENCH --secs 5 > /tmp/pcie_events.log 2>&1 &
    BENCH_PID=$!
    
    sleep 1
    echo ""
    echo "5. Monitoring PCIe status during kernel launches..."
    for i in {1..5}; do
        STATUS=$(read_pcie_status)
        echo "   [$i] $STATUS"
        sleep 1
    done
    
    wait $BENCH_PID 2>/dev/null || true
    echo "   ✓ Event test complete"
else
    echo "   (Skipping event_overhead test - not built)"
fi
echo ""

# Check final idle state
sleep 1
echo ""
echo "6. PCIe status after load (idle again):"
FINAL_STATUS=$(read_pcie_status)
echo "   $FINAL_STATUS"
echo ""

# Analysis
echo "========================================================================"
echo "ANALYSIS"
echo "========================================================================"
echo "Idle (before):  $IDLE_STATUS"
echo "Under load:     (see samples above)"
echo "Idle (after):   $FINAL_STATUS"
echo ""

# Extract speed from status strings
IDLE_SPEED=$(echo "$IDLE_STATUS" | grep -oP "[\d.]+ GT/s" | grep -oP "[\d.]+")
FINAL_SPEED=$(echo "$FINAL_STATUS" | grep -oP "[\d.]+ GT/s" | grep -oP "[\d.]+")

if [ "$IDLE_SPEED" == "$FINAL_SPEED" ]; then
    echo "Result: PCIe link speed did NOT change during load"
    echo ""
    if [[ "$IDLE_SPEED" == "2.5" ]]; then
        echo "✗ PROBLEM CONFIRMED: PCIe is stuck at Gen1 even under load"
        echo "  This is NOT power management - it's a configuration issue"
        echo ""
        echo "  Required BIOS changes:"
        echo "    - Enable 'Above 4G Decoding'"
        echo "    - Enable 'Resizable BAR'"
        echo "    - Verify GPU is in x16 slot"
        echo "    - Disable PCIe power management (ASPM) if enabled"
    else
        echo "✓ PCIe is stable but may still need optimization"
    fi
else
    echo "Result: PCIe link speed CHANGED during load"
    echo "  This suggests power management (ASPM) is active"
    echo "  Consider disabling ASPM for consistent performance"
fi

echo ""
echo "========================================================================"
echo "BENCHMARK RESULTS"
echo "========================================================================"
echo ""
echo "Batch transfers (sustained load):"
if [ -f /tmp/pcie_batch.log ]; then
    grep "GiB/s" /tmp/pcie_batch.log | tail -3
fi
echo ""
echo "Size sweep results:"
if [ -f /tmp/pcie_sweep.log ]; then
    echo "  (Full results in /tmp/pcie_sweep.log)"
    grep "GiB/s" /tmp/pcie_sweep.log | tail -5
fi
echo ""
if [ -f /tmp/pcie_events.log ]; then
    echo "Event overhead:"
    grep -E "avg|median" /tmp/pcie_events.log | head -5
fi
echo ""
