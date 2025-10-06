# Requirements Document

## Introduction

This document outlines the requirements for fixing the oneAPI backend implementation for Celerity to properly support Intel Arc GPUs using the Level Zero backend through SYCL/DPC++. The current implementation has critical issues causing infinite loops and test failures. The backend needs to be refactored to follow the same patterns as the CUDA backend while properly utilizing Level Zero's capabilities.

## Requirements

### Requirement 1: Backend Architecture Alignment

**User Story:** As a Celerity developer, I want the oneAPI backend to follow the same architectural patterns as the CUDA backend, so that it integrates properly with the Celerity runtime and is maintainable.

#### Acceptance Criteria

1. WHEN the oneAPI backend is instantiated THEN it SHALL inherit from `sycl_backend` base class like `sycl_cuda_backend` does
2. WHEN the backend is constructed THEN it SHALL use SYCL device and context management provided by the base class
3. WHEN device operations are enqueued THEN they SHALL use the `enqueue_device_work` pattern from the base class
4. IF the backend needs Level Zero specific operations THEN it SHALL access native handles through SYCL interop APIs

### Requirement 2: Memory Copy Operations

**User Story:** As a Celerity runtime, I want efficient n-dimensional memory copy operations on Intel Arc GPUs, so that data transfers don't become a performance bottleneck.

#### Acceptance Criteria

1. WHEN a 1D contiguous copy is requested THEN the backend SHALL use SYCL queue memcpy operations
2. WHEN a 2D or 3D strided copy is requested THEN the backend SHALL use optimized copy mechanisms
3. IF SYCL provides native 2D/3D copy operations THEN they SHALL be used
4. IF native operations are unavailable THEN the backend SHALL fall back to element-wise parallel_for with proper synchronization
5. WHEN any copy operation completes THEN it SHALL return a valid async_event that can be queried for completion

### Requirement 3: Event Handling and Synchronization

**User Story:** As a Celerity executor, I want reliable event-based synchronization for oneAPI operations, so that I can track operation completion and avoid infinite loops.

#### Acceptance Criteria

1. WHEN an asynchronous operation is enqueued THEN it SHALL return an async_event wrapping a SYCL event
2. WHEN check_async_errors is called THEN it SHALL only check for exceptions without blocking or spinning
3. WHEN an event is queried for completion THEN it SHALL use SYCL event query mechanisms
4. IF profiling is enabled THEN events SHALL capture timing information

### Requirement 4: Device Selection and Initialization

**User Story:** As a Celerity user, I want the oneAPI backend to properly initialize Intel Arc GPUs, so that I can run Celerity applications on Level Zero devices.

#### Acceptance Criteria

1. WHEN devices are selected THEN the backend SHALL accept SYCL devices compatible with Level Zero
2. WHEN the backend initializes THEN it SHALL properly configure SYCL queues with in-order execution
3. WHEN profiling is disabled THEN queues SHALL use the discard_events property to minimize overhead
4. WHEN multiple devices are present THEN peer-to-peer copy capabilities SHALL be properly detected and configured

### Requirement 5: Integration with Existing Codebase

**User Story:** As a Celerity maintainer, I want the oneAPI backend to integrate seamlessly with existing backend selection and device management code, so that users can easily switch between backends.

#### Acceptance Criteria

1. WHEN the backend is created THEN it SHALL be instantiated through the `make_sycl_backend` factory function
2. WHEN backend type is queried THEN it SHALL be identified as `sycl_backend_type::level_zero`
3. WHEN system_info is requested THEN it SHALL provide accurate device capabilities and memory topology
4. WHEN the backend is destroyed THEN all resources SHALL be properly cleaned up without leaks

### Requirement 6: Error Handling and Diagnostics

**User Story:** As a Celerity developer debugging issues, I want clear error messages and proper exception handling, so that I can quickly identify and fix problems.

#### Acceptance Criteria

1. WHEN a SYCL operation fails THEN the backend SHALL propagate the exception with context information
2. WHEN Level Zero operations are used THEN errors SHALL be checked and reported with meaningful messages
3. WHEN async errors occur THEN they SHALL be captured by the async_handler and logged
4. WHEN check_async_errors is called THEN it SHALL call throw_asynchronous on SYCL queues to surface any pending errors
