#!/bin/bash
#SBATCH --job-name=all_tests_test
#SBATCH --partition=intel
#SBATCH --gres=gpu:ARC770:1
#SBATCH --time=00:10:00
#SBATCH --output=output.log
#SBATCH --error=error.log

echo "SLURM job running on $(hostname)"
echo "Job started at $(date)"

# Load oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Set SYCL device filter to select the desired GPU
export ONEAPI_DEVICE_SELECTOR="level_zero:*" 

# Navigate to your build directory
cd ~/testApproach/celerity-runtime/currentBuild/test/

# Run
echo "backend_tests"
./backend_tests
#echo "all_tests"
#./all_tests
