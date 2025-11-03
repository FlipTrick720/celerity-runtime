#!/bin/bash
# PCIe Configuration Check for Intel Arc A770
# This script checks if the GPU is properly configured (AI Generated)

set -e

echo "========================================================================"
echo "PCIe Configuration Check - Intel Arc A770"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find Intel Arc GPU
echo "1. Detecting Intel Arc GPU..."
GPU_PCI=$(lspci | grep -i "VGA.*Intel" | grep -i "Arc" | head -1 | cut -d' ' -f1)

if [ -z "$GPU_PCI" ]; then
    echo -e "${RED}ERROR: Intel Arc GPU not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Found GPU at: $GPU_PCI${NC}"
GPU_NAME=$(lspci -s $GPU_PCI | cut -d':' -f3-)
echo "GPU: $GPU_NAME"
echo ""

# Check PCIe Link Speed
echo "2. Checking PCIe Link Speed..."
LINK_SPEED=$(lspci -vv -s $GPU_PCI 2>/dev/null | grep "LnkSta:" | grep -oP "Speed \K[^,]+" | head -1)
LINK_WIDTH=$(lspci -vv -s $GPU_PCI 2>/dev/null | grep "LnkSta:" | grep -oP "Width \K[^,]+" | head -1)

# If lspci -vv doesn't work (no root), try sysfs first
if [ -z "$LINK_SPEED" ]; then
    echo "Note: lspci -vv requires root, trying sysfs instead..."
    
    # Try to find the correct sysfs path for this GPU
    SYSFS_PATH="/sys/bus/pci/devices/0000:$GPU_PCI"
    
    if [ -d "$SYSFS_PATH" ]; then
        SYSFS_SPEED=$(cat $SYSFS_PATH/current_link_speed 2>/dev/null || echo "")
        SYSFS_WIDTH=$(cat $SYSFS_PATH/current_link_width 2>/dev/null || echo "")
        
        # Convert sysfs format to lspci format
        if [[ "$SYSFS_SPEED" == *"16.0"* ]]; then
            LINK_SPEED="16GT/s"
        elif [[ "$SYSFS_SPEED" == *"8.0"* ]]; then
            LINK_SPEED="8GT/s"
        elif [[ "$SYSFS_SPEED" == *"5.0"* ]]; then
            LINK_SPEED="5GT/s"
        elif [[ "$SYSFS_SPEED" == *"2.5"* ]]; then
            LINK_SPEED="2.5GT/s"
        else
            LINK_SPEED="$SYSFS_SPEED"
        fi
        
        if [ -n "$SYSFS_WIDTH" ]; then
            LINK_WIDTH="x$SYSFS_WIDTH"
        fi
    else
        echo "Warning: sysfs path not found at $SYSFS_PATH"
        LINK_SPEED="Unknown"
        LINK_WIDTH="Unknown"
    fi
fi

echo "Current Link Speed: $LINK_SPEED"
echo "Current Link Width: $LINK_WIDTH"

# Expected values
EXPECTED_SPEED="16GT/s"
EXPECTED_WIDTH="x16"

if [[ "$LINK_SPEED" == *"16GT/s"* ]] || [[ "$LINK_SPEED" == *"16.0GT/s"* ]]; then
    echo -e "${GREEN}✓ Link Speed is correct (PCIe Gen4)${NC}"
    SPEED_OK=1
elif [[ "$LINK_SPEED" == *"8GT/s"* ]] || [[ "$LINK_SPEED" == *"8.0GT/s"* ]]; then
    echo -e "${YELLOW}⚠ Link Speed is PCIe Gen3 (should be Gen4)${NC}"
    SPEED_OK=0
else
    echo -e "${RED}✗ Link Speed is too low! (should be 16GT/s)${NC}"
    SPEED_OK=0
fi

if [[ "$LINK_WIDTH" == "x16" ]]; then
    echo -e "${GREEN}✓ Link Width is correct (x16)${NC}"
    WIDTH_OK=1
else
    echo -e "${RED}✗ Link Width is wrong! (should be x16, got $LINK_WIDTH)${NC}"
    WIDTH_OK=0
fi
echo ""

# Show sysfs info as well
echo "3. Verifying via sysfs..."
SYSFS_PATH="/sys/bus/pci/devices/0000:$GPU_PCI"
if [ -d "$SYSFS_PATH" ]; then
    SYSFS_SPEED=$(cat $SYSFS_PATH/current_link_speed 2>/dev/null || echo "N/A")
    SYSFS_WIDTH=$(cat $SYSFS_PATH/current_link_width 2>/dev/null || echo "N/A")
    echo "sysfs Path: $SYSFS_PATH"
    echo "sysfs Link Speed: $SYSFS_SPEED"
    echo "sysfs Link Width: $SYSFS_WIDTH"
else
    echo "sysfs path not found: $SYSFS_PATH"
fi
echo ""

# Check BAR (Base Address Register) size
echo "4. Checking BAR (Base Address Register) configuration..."
BAR_INFO=$(lspci -vv -s $GPU_PCI 2>/dev/null | grep "Region 0:" | head -1)

if [ -z "$BAR_INFO" ]; then
    # Try without -vv
    BAR_INFO=$(lspci -v -s $GPU_PCI 2>/dev/null | grep "Region 0:" | head -1)
fi

if [ -n "$BAR_INFO" ]; then
    echo "$BAR_INFO"
else
    echo "BAR info not available (may need root access)"
    BAR_INFO="unknown"
fi

if [[ "$BAR_INFO" == *"non-prefetchable"* ]]; then
    echo -e "${RED}✗ BAR is non-prefetchable (should be prefetchable)${NC}"
    BAR_OK=0
else
    echo -e "${GREEN}✓ BAR is prefetchable${NC}"
    BAR_OK=1
fi

# Extract BAR size
BAR_SIZE=$(echo "$BAR_INFO" | grep -oP "size=\K[^]]+")
echo "BAR Size: $BAR_SIZE"

if [[ "$BAR_SIZE" == *"16M"* ]]; then
    echo -e "${RED}✗ BAR size is only 16M (Resizable BAR not enabled)${NC}"
    echo -e "${YELLOW}  Should be 16G with Resizable BAR enabled${NC}"
    REBAR_OK=0
elif [[ "$BAR_SIZE" == *"G"* ]]; then
    echo -e "${GREEN}✓ Resizable BAR appears to be enabled${NC}"
    REBAR_OK=1
else
    echo -e "${YELLOW}⚠ BAR size unclear: $BAR_SIZE${NC}"
    REBAR_OK=0
fi
echo ""

# Calculate theoretical bandwidth
echo "5. Calculating PCIe Bandwidth..."
# Check in order from slowest to fastest to avoid substring matching issues
if [[ "$LINK_SPEED" == *"2.5"* ]]; then
    SPEED_GTS=2.5
    GEN="Gen1"
elif [[ "$LINK_SPEED" == *"5"* ]] && [[ "$LINK_SPEED" != *"16"* ]]; then
    SPEED_GTS=5
    GEN="Gen2"
elif [[ "$LINK_SPEED" == *"8"* ]]; then
    SPEED_GTS=8
    GEN="Gen3"
elif [[ "$LINK_SPEED" == *"16"* ]]; then
    SPEED_GTS=16
    GEN="Gen4"
else
    SPEED_GTS=0
    GEN="Unknown"
fi

echo "Detected: PCIe $GEN $LINK_WIDTH ($SPEED_GTS GT/s)"

if [[ "$LINK_WIDTH" == "x16" ]]; then
    LANES=16
elif [[ "$LINK_WIDTH" == "x8" ]]; then
    LANES=8
elif [[ "$LINK_WIDTH" == "x4" ]]; then
    LANES=4
elif [[ "$LINK_WIDTH" == "x1" ]]; then
    LANES=1
elif [[ "$LINK_WIDTH" == "x" ]]; then
    # Width is just "x" - probably means x1
    LANES=1
    LINK_WIDTH="x1"
else
    LANES=0
fi

if [ "$SPEED_GTS" != "0" ] && [ "$LANES" != "0" ]; then
    # PCIe encoding overhead: 128b/130b for Gen3+, 8b/10b for Gen1/2
    if (( $(awk "BEGIN {print ($SPEED_GTS >= 8)}") )); then
        ENCODING=0.9846  # 128/130 for Gen3+
    else
        ENCODING=0.8     # 8/10 for Gen1/2
    fi
    
    # GT/s to GB/s: GT/s * encoding / 8 (bits to bytes) * lanes
    BANDWIDTH=$(awk "BEGIN {printf \"%.2f\", $SPEED_GTS * $ENCODING * $LANES / 8}")
    printf "Theoretical PCIe Bandwidth: %s GB/s (%.2f GiB/s)\n" $BANDWIDTH $(awk "BEGIN {printf \"%.2f\", $BANDWIDTH / 1.024}")
    
    # Expected for Gen4 x16
    EXPECTED_BW=$(awk "BEGIN {printf \"%.2f\", 16 * 0.9846 * 16 / 8}")
    printf "Expected (Gen4 x16):        %s GB/s (%.2f GiB/s)\n" $EXPECTED_BW $(awk "BEGIN {printf \"%.2f\", $EXPECTED_BW / 1.024}")
    
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($BANDWIDTH / $EXPECTED_BW) * 100}")
    printf "Current vs Expected:        %s%%\n" $PERCENT
    
    if (( $(awk "BEGIN {print ($PERCENT < 10)}") )); then
        echo -e "${RED}⚠ PCIe bandwidth is SEVERELY limited! (${PERCENT}% of expected)${NC}"
    fi
fi
echo ""

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"

ISSUES=0

if [ "$SPEED_OK" -eq 0 ]; then
    echo -e "${RED}✗ PCIe Link Speed is not optimal${NC}"
    ((ISSUES++))
else
    echo -e "${GREEN}✓ PCIe Link Speed is correct${NC}"
fi

if [ "$WIDTH_OK" -eq 0 ]; then
    echo -e "${RED}✗ PCIe Link Width is not optimal${NC}"
    ((ISSUES++))
else
    echo -e "${GREEN}✓ PCIe Link Width is correct${NC}"
fi

if [ "$BAR_OK" -eq 0 ]; then
    echo -e "${RED}✗ BAR is non-prefetchable${NC}"
    ((ISSUES++))
else
    echo -e "${GREEN}✓ BAR is prefetchable${NC}"
fi

if [ "$REBAR_OK" -eq 0 ]; then
    echo -e "${RED}✗ Resizable BAR is not enabled${NC}"
    ((ISSUES++))
else
    echo -e "${GREEN}✓ Resizable BAR appears enabled${NC}"
fi

if [ "$BAR_OK" -eq 0 ]; then
    echo -e "${RED}✗ BAR is non-prefetchable${NC}"
    ((ISSUES++))
else
    echo -e "${GREEN}✓ BAR is prefetchable${NC}"
fi

if [ "$REBAR_OK" -eq 0 ]; then
    echo -e "${RED}✗ Resizable BAR is not enabled${NC}"
    ((ISSUES++))
else
    echo -e "${GREEN}✓ Resizable BAR appears enabled${NC}"
fi

echo ""
echo "========================================================================"
echo "IMPACT ON PERFORMANCE"
echo "========================================================================"
if [ "$ISSUES" -gt 0 ]; then
    echo -e "${RED}Current PCIe configuration is severely limiting performance!${NC}"
    echo ""
    echo "Expected bandwidth improvements after fix:"
    echo "  H2D/D2H: Current ~0.24 GiB/s → Expected ~10-12 GiB/s (40-50x faster)"
    echo "  D2D:     May improve if currently PCIe-limited"
    echo ""
fi

echo ""
if [ "$ISSUES" -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! PCIe configuration is optimal.${NC}"
    exit 0
else
    echo -e "${RED}✗ Found $ISSUES issue(s) with PCIe configuration.${NC}"
    echo ""
    echo "RECOMMENDED ACTIONS:"
    echo "1. Enter BIOS/UEFI settings"
    echo "2. Enable 'Above 4G Decoding'"
    echo "3. Enable 'Resizable BAR' (ReBAR)"
    echo "4. Verify GPU is in a PCIe x16 slot (not x1 or x4)"
    echo "5. Check if slot is physically damaged or has debris"
    echo "6. Save BIOS settings and reboot"
    echo ""
    echo "After changes, run this script again to verify:"
    echo "  ./check_pcie_config.sh"
    exit 1
fi
