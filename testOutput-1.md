malte.braig@gpuc5:~/testApproach/celerity-runtime$ ./run_test.sh
SLURM job running on gpuc5
Job started at Wed Aug 13 17:15:15 UTC 2025
 
:: initializing oneAPI environment ...
   run_test.sh: BASH_VERSION = 5.2.21(1)-release
   args: Using "$@" for setvars.sh arguments: 
:: ccl -- latest
:: compiler -- latest
:: dal -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: dpcpp-ct -- latest
:: dpl -- latest
:: ipp -- latest
:: ippcp -- latest
:: mkl -- latest
:: mpi -- latest
:: pti -- latest
:: tbb -- latest
:: umf -- latest
:: vtune -- latest
:: oneAPI environment initialized ::
 
all_tests
Randomness seeded to: 124954948

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
all_tests is a Catch2 v3.5.0 host application.
Run with -? for options

-------------------------------------------------------------------------------
device accessor reports out-of-bounds accesses - 1
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:665
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:667: SKIPPED:
explicitly with message:
  CELERITY_ACCESSOR_BOUNDARY_CHECK=0

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:307: warning:
  Test specified a runtime_fixture, but did not end up instantiating the
  runtime

-------------------------------------------------------------------------------
device accessor reports out-of-bounds accesses - 2
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:665
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:667: SKIPPED:
explicitly with message:
  CELERITY_ACCESSOR_BOUNDARY_CHECK=0

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:307: warning:
  Test specified a runtime_fixture, but did not end up instantiating the
  runtime

-------------------------------------------------------------------------------
device accessor reports out-of-bounds accesses - 3
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:665
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:667: SKIPPED:
explicitly with message:
  CELERITY_ACCESSOR_BOUNDARY_CHECK=0

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:307: warning:
  Test specified a runtime_fixture, but did not end up instantiating the
  runtime

-------------------------------------------------------------------------------
host accessor reports out-of-bounds accesses - 1
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:712
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:714: SKIPPED:
explicitly with message:
  CELERITY_ACCESSOR_BOUNDARY_CHECK=0

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:307: warning:
  Test specified a runtime_fixture, but did not end up instantiating the
  runtime

-------------------------------------------------------------------------------
host accessor reports out-of-bounds accesses - 2
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:712
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:714: SKIPPED:
explicitly with message:
  CELERITY_ACCESSOR_BOUNDARY_CHECK=0

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:307: warning:
  Test specified a runtime_fixture, but did not end up instantiating the
  runtime

-------------------------------------------------------------------------------
host accessor reports out-of-bounds accesses - 3
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:712
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/accessor_tests.cc:714: SKIPPED:
explicitly with message:
  CELERITY_ACCESSOR_BOUNDARY_CHECK=0

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:307: warning:
  Test specified a runtime_fixture, but did not end up instantiating the
  runtime

-------------------------------------------------------------------------------
backend allocations are pattern-filled in debug builds
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/backend_tests.cc:116
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/backend_tests.cc:137: SKIPPED:
explicitly with message:
  Not in a debug build

-------------------------------------------------------------------------------
backend copies work correctly on all source- and destination layouts
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/backend_tests.cc:313
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/backend_tests.cc:340: SKIPPED:
explicitly with messages:
  Available devices do not support peer-to-peer copy

-------------------------------------------------------------------------------
handler throws when accessor target does not match command type
  capturing host accessor into device kernel
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/runtime_tests.cc:368
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/runtime_tests.cc:382: SKIPPED:
explicitly with message:
  DPC++ does not allow for host accessors to be captured into kernels.

-------------------------------------------------------------------------------
handler throws when accessing host objects within device tasks
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/runtime_tests.cc:401
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/runtime_tests.cc:414: SKIPPED:
explicitly with message:
  DPC++ does not allow for side effects to be captured into kernels.

-------------------------------------------------------------------------------
handler recognizes copies of same side-effect being captured multiple times
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/runtime_tests.cc:506
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/runtime_tests.cc:524: FAILED:
  CHECK_THROWS_WITH( ([&] { q.submit([&](handler& cgh) { experimental::side_effect se1{ho1, cgh}; auto se2 = se1; experimental::side_effect se3{ho2, cgh}; cgh.host_task(on_master_node, [se1, se2, &se3 ]() { (void)se1; (void)se2; (void)se3; }); }); })(), "(NYI)" )
because no exception was thrown where one was expected:

captured log:
  [info] Celerity runtime version 0.6.0 6b90ee3-dirty running on DPC++ / Clang 19.0.0git (icx 2025.0.4.20241205) / Open MPI v4.1.6. PID = 1860766, build type = release, using the default allocator
  [info] Using platform "Intel(R) oneAPI Unified Runtime over Level-Zero", device "Intel(R) Arc(TM) A770 Graphics" as D0 (automatically selected)
  [info] Using platform "Intel(R) oneAPI Unified Runtime over Level-Zero", device "Intel(R) Arc(TM) A770 Graphics" as D1 (automatically selected)
  [trace] Affinity: Initialized, available cores: 111111111111111111111111111111111111111111111111
  [debug] Affinity: pinned thread 'cy-application' to core 1 (local process #0 thread id 7daec8cc3180)
  [warning] No common backend specialization available for all selected devices, falling back to generic. Performance may be degraded.
  [debug] Generic backend does not support peer memory access, device-to-device copies will be staged in host memory
  [trace] Horizon decision: false - seq: false para: false - crit_p: 0 exec_f: 1
  [trace] Affinity: thread 'cy-alloc' is not pinned.
  [debug] Affinity: pinned thread 'cy-dev-sub-0' to core 4 (local process #0 thread id 7daea9e006c0)
  [debug] Affinity: pinned thread 'cy-dev-sub-1' to core 5 (local process #0 thread id 7daea8a006c0)
  [debug] Affinity: pinned thread 'cy-executor' to core 3 (local process #0 thread id 7daea34006c0)
  [debug] Affinity: pinned thread 'cy-scheduler' to core 2 (local process #0 thread id 7daea94006c0)
  [trace] Scheduler is busy
  [trace] [executor] I0: epoch (init)
  [trace] [executor] I1: host task, [0,0,0] - [1,1,1]
  [trace] Scheduler is idle
  [trace] Affinity: thread 'cy-host-0' is not pinned.
  [trace] [executor] retired I1
  [trace] [executor] I2: destroy H1
  [trace] [executor] I3: destroy H0
  [trace] [executor] I4: epoch
  [trace] Scheduler is busy
  [trace] Scheduler is idle
  [trace] [executor] I5: epoch (shutdown)
  [trace] Scheduler is busy
  [debug] Executor active time: 3.6 ms. Starvation time: 63.6 µs (1.8%).

-------------------------------------------------------------------------------
tests that log any message in excess of level::info fail by default
-------------------------------------------------------------------------------
/home/malte.braig/testApproach/celerity-runtime/test/test_utils_tests.cc:13
...............................................................................

/home/malte.braig/testApproach/celerity-runtime/test/test_utils.cc:74: FAILED:
explicitly with message:
  Observed a log message exceeding the expected maximum of "info". If this is
  correct, increase the expected log level through test_utils::
  allow_max_log_level().

captured log:
  [warning] spooky message!

===============================================================================
test cases:   459 |   447 passed | 10 skipped | 2 failed as expected
assertions: 29687 | 29685 passed |  0 skipped | 2 failed as expected