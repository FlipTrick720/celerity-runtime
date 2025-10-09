# Solution Summary: Affinity Test Failures

## Problem

Your test environment has **insufficient CPU cores** available for thread pinning:
- Available: Only 2 cores (cores 0 and 24)
- Required: 4 cores (application, scheduler, executor, device submitter threads)

This caused two cascading issues:
1. The affinity warning was logged at `info` level instead of `warn` level
2. After fixing #1, ALL tests failed because they unexpectedly received a warning

## Root Cause

The test environment uses `ZE_AFFINITY_MASK=0` which restricts the process to specific cores. With only 2 cores available but 4 needed, the runtime correctly warns about insufficient cores. However:

1. The warning was logged at the wrong level (`INFO` instead of `WARN`)
2. The test framework wasn't configured to expect this system-dependent warning

## Solution

Two files were modified:

### 1. src/platform_specific/affinity.unix.cc (line 148)
Changed the log level from `CELERITY_INFO` to `CELERITY_WARN`:

```cpp
CELERITY_WARN("Insufficient logical cores available for thread pinning (required {} starting from {}, {} available), disabling pinning."
              " Performance may be negatively impacted.", //
    total_threads, cfg.standard_core_start_id, CPU_COUNT(&g_state.available_cores));
```

**Why**: This is semantically correct - insufficient resources that impact performance should be a warning, not just info.

### 2. test/test_utils.cc (line 244)
Added the affinity warning to the expected warnings regex:

```cpp
const char* const expected_runtime_init_warnings_regex = 
    "Celerity has detected that only .* logical cores are available to this process.*|"
    "Insufficient logical cores available for thread pinning.*|"  // <-- ADDED THIS LINE
    "Celerity detected more than one node \\(MPI rank\\) on this host, which is not recommended.*|"
    "Instrumentation for profiling with Tracy is enabled\\. Performance may be negatively impacted\\.|";
```

**Why**: This allows tests to pass even when the system has insufficient cores, which is a system constraint, not a test failure.

## How It Works

The `runtime_fixture` (used by most tests) automatically allows certain system-dependent warnings that don't indicate test failures. By adding the affinity warning to this list:

1. Tests that specifically check for this warning (like `affinity_tests.cc:148`) still work correctly
2. All other tests don't fail just because the system has insufficient cores
3. The warning is still logged (visible in test output) but doesn't cause test failures

## Expected Test Results

After these changes, you should see:
```
test cases:   459 |   447 passed | 0 failed | 10 skipped | 2 failed as expected
assertions: 32942 | 32940 passed | 0 failed |  0 skipped | 2 failed as expected
```

The 2 "failed as expected" are intentional meta-tests marked with `[!shouldfail]` that verify the test framework correctly catches unexpected warnings.

## Why This Approach is Correct

1. **Semantically correct**: Insufficient resources that impact performance should be warnings
2. **Follows existing patterns**: Other system-dependent warnings (like "only X cores available") are already handled this way
3. **Doesn't hide real issues**: The warning is still logged and visible
4. **Environment-aware**: Tests don't fail due to system constraints beyond the test's control
5. **Maintains test coverage**: The specific affinity test still validates the warning is emitted

## Files Modified

1. `src/platform_specific/affinity.unix.cc` - Changed log level from INFO to WARN
2. `test/test_utils.cc` - Added affinity warning to expected warnings regex
