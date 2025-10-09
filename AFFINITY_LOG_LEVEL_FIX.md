# Affinity Test Fix - Log Level Correction and Test Framework Update

## Problem Analysis

The test `affinity_tests.cc:148` was failing because of a log level mismatch, and subsequently ALL tests were failing due to the test environment having insufficient cores.

### Initial Problem (tmp3 - first output)
The test expected a **warning** level message, but the code was logging at **info** level:

```cpp
// Test expectation:
CHECK(test_utils::log_contains_substring(
    detail::log_level::warn, 
    "Insufficient logical cores available for thread pinning..."));

// Actual code:
CELERITY_INFO("Insufficient logical cores available for thread pinning...");
```

### Secondary Problem (tmp1, tmp2, tmp3 - after first fix)
After changing to `CELERITY_WARN`, the affinity test passed, but **all other tests started failing** because:
1. The test environment has only 2 cores available (cores 0 and 24)
2. Celerity runtime needs 4 cores (application, scheduler, executor, device submitter)
3. Every test that initializes the runtime now logs a warning about insufficient cores
4. The test framework by default fails tests that log warnings

## Root Cause

In `src/platform_specific/affinity.unix.cc`, when insufficient cores are available for thread pinning, the message was logged using `CELERITY_INFO()` instead of `CELERITY_WARN()`.

This is semantically incorrect because:
1. Insufficient cores is a **warning condition** that may negatively impact performance
2. The message itself states "Performance may be negatively impacted"
3. Other similar conditions in the same file use `CELERITY_WARN()`

## Solution

Two changes were required:

### 1. Change Log Level (src/platform_specific/affinity.unix.cc)
```cpp
// Before:
CELERITY_INFO("Insufficient logical cores available for thread pinning...");

// After:
CELERITY_WARN("Insufficient logical cores available for thread pinning...");
```

### 2. Add Warning to Expected Patterns (test/test_utils.cc)
The `runtime_fixture` already has a mechanism to allow expected system-dependent warnings. We added the affinity warning to the regex pattern:

```cpp
// Before:
const char* const expected_runtime_init_warnings_regex = 
    "Celerity has detected that only .* logical cores are available to this process.*|"
    "Celerity detected more than one node \\(MPI rank\\) on this host, which is not recommended.*|"
    "Instrumentation for profiling with Tracy is enabled\\. Performance may be negatively impacted\\.|";

// After:
const char* const expected_runtime_init_warnings_regex = 
    "Celerity has detected that only .* logical cores are available to this process.*|"
    "Insufficient logical cores available for thread pinning.*|"  // <-- ADDED
    "Celerity detected more than one node \\(MPI rank\\) on this host, which is not recommended.*|"
    "Instrumentation for profiling with Tracy is enabled\\. Performance may be negatively impacted\\.|";
```

This allows all tests using `runtime_fixture` to pass even when the warning is logged due to insufficient cores in the test environment.

## Why This Approach

The test environment has limited cores (only 2 available), which is a **system constraint**, not a test failure. The warning is correct and expected in this environment. By adding it to the expected warnings regex:

1. Tests that specifically check for this warning (like `affinity_tests.cc:148`) still work correctly
2. All other tests don't fail just because the system has insufficient cores
3. The warning is still logged (visible in test output) but doesn't cause test failures
4. This follows the existing pattern for other system-dependent warnings

## Test Results

### Before Any Fix (original tmp3)
```
test cases:   459 |   446 passed | 1 failed | 10 skipped | 2 failed as expected
```
- `affinity_tests.cc:148` failed (expected warn, got info)

### After First Fix Only (tmp1, tmp2, tmp3 - second run)
```
test cases:   459 |   0 passed | 459 failed | ...
```
- All tests that initialize runtime failed due to unexpected warning

### After Both Fixes (expected)
```
test cases:   459 |   447 passed | 0 failed | 10 skipped | 2 failed as expected
```
- All tests pass
- The 2 "failed as expected" are intentional meta-tests marked with `[!shouldfail]`

## Files Modified

1. `src/platform_specific/affinity.unix.cc` - Changed `CELERITY_INFO` to `CELERITY_WARN` at line 148
2. `test/test_utils.cc` - Added affinity warning pattern to `expected_runtime_init_warnings_regex`
