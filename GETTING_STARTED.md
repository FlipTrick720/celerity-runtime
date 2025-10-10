# Getting Started with Celerity Runtime

Quick guide to clone, build, and test the Celerity runtime with oneAPI Level-Zero backend.

## Prerequisites

- Intel oneAPI Toolkit (DPC++)
- MPI (Open MPI or Intel MPI)
- Ninja build system
- Git
- Level-Zero drivers for Intel GPUs

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/FlipTrick720/celerity-runtime.git
cd celerity-runtime
git checkout optimized-setup-test
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Make Scripts Executable

```bash
chmod +x build_and_test.sh
chmod +x build_celerity.sh
chmod +x run_test.sh
```

### 4. Build and Test

Choose one of the following options:

#### Option A: Build and Test Together (Recommended)

```bash
./build_and_test.sh
```

This will:
- Build the project
- Submit tests via SLURM
- Save logs to `z_output.log`

#### Option B: Build Only

```bash
./build_celerity.sh
```

This will:
- Configure with CMake
- Build with Ninja
- Save logs to `build_logs/build_<timestamp>.log`

#### Option C: Test Only (After Building)

```bash
sbatch run_test.sh --profile test
```

This will:
- Find the newest build directory
- Run all tests with the specified profile
- Save logs to `build_test_logs/run_<timestamp>.log`

## Directory Structure After Setup

```
celerity-runtime/
├── build_and_test.sh          # Build + test script
├── build_celerity.sh           # Build-only script
├── run_test.sh                 # Test runner script
├── build_logs/                 # Build logs
│   └── build_<timestamp>.log
├── build_test_logs/            # Test logs
│   └── run_<timestamp>.log
├── build_<timestamp>/          # Build directories
│   ├── test/
│   │   └── all_tests          # Test executable
│   └── ...
├── vendor/                     # Git submodules
│   ├── Catch2/
│   ├── spdlog/
│   └── libenvpp/
└── ...
```

## Test Profiles

The test runner supports different profiles for various use cases:

| Profile | Command | Use Case |
|---------|---------|----------|
| `test` | `./run_test.sh --profile test` | **Recommended** - CI-like testing with validation |
| `clean` | `./run_test.sh --profile clean` | Minimal logging for normal runs |
| `verbose` | `./run_test.sh --profile verbose` | Detailed UR tracing for debugging |
| `debug` | `./run_test.sh --profile debug` | Maximum debugging with auto-GDB |
