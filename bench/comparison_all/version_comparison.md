# Backend Version Comparison

| Version | Operation | Mode | Pinned | Peak (GiB/s) | Median (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| v0_baseline | D2D | Sync | Yes | 188.83 | 5.95 |
| v0_baseline | D2D | Sync | No | 189.61 | 7.20 |
| v0_baseline | D2D | Batch | Yes | 208.54 | 41.15 |
| v0_baseline | D2D | Batch | No | 208.48 | 41.46 |
| v0_baseline | H2D | Sync | Yes | 10.23 | 4.13 |
| v0_baseline | H2D | Sync | No | 7.81 | 1.92 |
| v0_baseline | H2D | Batch | Yes | 10.28 | 8.21 |
| v0_baseline | H2D | Batch | No | 10.26 | 8.04 |
| v0_baseline | D2H | Sync | Yes | 11.04 | 4.37 |
| v0_baseline | D2H | Sync | No | 8.25 | 1.96 |
| v0_baseline | D2H | Batch | Yes | 11.10 | 9.52 |
| v0_baseline | D2H | Batch | No | 11.06 | 9.31 |
| v1_event_pooling | D2D | Sync | Yes | 181.13 | 2.52 |
| v1_event_pooling | D2D | Sync | No | 189.49 | 7.13 |
| v1_event_pooling | D2D | Batch | Yes | 208.52 | 41.41 |
| v1_event_pooling | D2D | Batch | No | 208.21 | 41.41 |
| v1_event_pooling | H2D | Sync | Yes | 10.23 | 4.12 |
| v1_event_pooling | H2D | Sync | No | 7.32 | 2.16 |
| v1_event_pooling | H2D | Batch | Yes | 10.28 | 8.22 |
| v1_event_pooling | H2D | Batch | No | 10.25 | 8.03 |
| v1_event_pooling | D2H | Sync | Yes | 11.04 | 4.37 |
| v1_event_pooling | D2H | Sync | No | 7.72 | 2.32 |
| v1_event_pooling | D2H | Batch | Yes | 11.10 | 9.52 |
| v1_event_pooling | D2H | Batch | No | 11.06 | 9.30 |
| v2_immediate_pool | D2D | Sync | Yes | 188.88 | 6.92 |
| v2_immediate_pool | D2D | Sync | No | 180.17 | 3.56 |
| v2_immediate_pool | D2D | Batch | Yes | 208.54 | 41.38 |
| v2_immediate_pool | D2D | Batch | No | 208.58 | 41.46 |
| v2_immediate_pool | H2D | Sync | Yes | 10.23 | 4.15 |
| v2_immediate_pool | H2D | Sync | No | 7.29 | 1.91 |
| v2_immediate_pool | H2D | Batch | Yes | 10.28 | 8.20 |
| v2_immediate_pool | H2D | Batch | No | 10.25 | 8.10 |
| v2_immediate_pool | D2H | Sync | Yes | 11.04 | 4.39 |
| v2_immediate_pool | D2H | Sync | No | 7.74 | 1.96 |
| v2_immediate_pool | D2H | Batch | Yes | 11.10 | 9.52 |
| v2_immediate_pool | D2H | Batch | No | 11.06 | 9.31 |
| v3_batch_fence | D2D | Sync | Yes | 188.72 | 5.97 |
| v3_batch_fence | D2D | Sync | No | 188.86 | 6.32 |
| v3_batch_fence | D2D | Batch | Yes | 208.56 | 41.31 |
| v3_batch_fence | D2D | Batch | No | 208.52 | 41.55 |
| v3_batch_fence | H2D | Sync | Yes | 10.23 | 4.13 |
| v3_batch_fence | H2D | Sync | No | 7.34 | 1.91 |
| v3_batch_fence | H2D | Batch | Yes | 10.28 | 8.22 |
| v3_batch_fence | H2D | Batch | No | 10.26 | 8.11 |
| v3_batch_fence | D2H | Sync | Yes | 11.04 | 4.37 |
| v3_batch_fence | D2H | Sync | No | 7.71 | 2.06 |
| v3_batch_fence | D2H | Batch | Yes | 11.09 | 9.52 |
| v3_batch_fence | D2H | Batch | No | 11.06 | 9.32 |
| v4_micro_optimized | D2D | Sync | Yes | 188.91 | 6.70 |
| v4_micro_optimized | D2D | Sync | No | 189.22 | 6.45 |
| v4_micro_optimized | D2D | Batch | Yes | 208.52 | 41.30 |
| v4_micro_optimized | D2D | Batch | No | 208.19 | 41.51 |
| v4_micro_optimized | H2D | Sync | Yes | 10.22 | 4.13 |
| v4_micro_optimized | H2D | Sync | No | 7.81 | 2.09 |
| v4_micro_optimized | H2D | Batch | Yes | 10.27 | 8.21 |
| v4_micro_optimized | H2D | Batch | No | 10.27 | 8.03 |
| v4_micro_optimized | D2H | Sync | Yes | 11.04 | 4.37 |
| v4_micro_optimized | D2H | Sync | No | 8.28 | 2.19 |
| v4_micro_optimized | D2H | Batch | Yes | 11.09 | 9.52 |
| v4_micro_optimized | D2H | Batch | No | 11.06 | 9.30 |
| v5_No_Sync | D2D | Sync | Yes | 178.95 | 1.96 |
| v5_No_Sync | D2D | Sync | No | 189.20 | 7.09 |
| v5_No_Sync | D2D | Batch | Yes | 208.46 | 41.23 |
| v5_No_Sync | D2D | Batch | No | 208.54 | 41.39 |
| v5_No_Sync | H2D | Sync | Yes | 10.22 | 4.13 |
| v5_No_Sync | H2D | Sync | No | 7.34 | 2.09 |
| v5_No_Sync | H2D | Batch | Yes | 10.28 | 8.23 |
| v5_No_Sync | H2D | Batch | No | 10.24 | 8.02 |
| v5_No_Sync | D2H | Sync | Yes | 11.04 | 4.37 |
| v5_No_Sync | D2H | Sync | No | 7.67 | 2.09 |
| v5_No_Sync | D2H | Batch | Yes | 11.10 | 9.52 |
| v5_No_Sync | D2H | Batch | No | 11.05 | 9.29 |
| v6_corrected_exact_fixes_async | D2D | Sync | Yes | 188.88 | 6.18 |
| v6_corrected_exact_fixes_async | D2D | Sync | No | 189.14 | 7.13 |
| v6_corrected_exact_fixes_async | D2D | Batch | Yes | 208.50 | 41.32 |
| v6_corrected_exact_fixes_async | D2D | Batch | No | 208.56 | 41.28 |
| v6_corrected_exact_fixes_async | H2D | Sync | Yes | 10.22 | 4.13 |
| v6_corrected_exact_fixes_async | H2D | Sync | No | 7.33 | 1.92 |
| v6_corrected_exact_fixes_async | H2D | Batch | Yes | 10.28 | 8.23 |
| v6_corrected_exact_fixes_async | H2D | Batch | No | 10.26 | 8.03 |
| v6_corrected_exact_fixes_async | D2H | Sync | Yes | 11.04 | 4.37 |
| v6_corrected_exact_fixes_async | D2H | Sync | No | 7.73 | 1.95 |
| v6_corrected_exact_fixes_async | D2H | Batch | Yes | 11.10 | 9.52 |
| v6_corrected_exact_fixes_async | D2H | Batch | No | 11.06 | 9.30 |
| v7_async_fence_optimized | D2D | Sync | Yes | 188.91 | 1.39 |
| v7_async_fence_optimized | D2D | Sync | No | 188.77 | 7.14 |
| v7_async_fence_optimized | D2D | Batch | Yes | 208.58 | 41.24 |
| v7_async_fence_optimized | D2D | Batch | No | 208.56 | 41.44 |
| v7_async_fence_optimized | H2D | Sync | Yes | 10.23 | 4.12 |
| v7_async_fence_optimized | H2D | Sync | No | 7.81 | 2.11 |
| v7_async_fence_optimized | H2D | Batch | Yes | 10.28 | 8.22 |
| v7_async_fence_optimized | H2D | Batch | No | 10.25 | 8.03 |
| v7_async_fence_optimized | D2H | Sync | Yes | 11.04 | 4.36 |
| v7_async_fence_optimized | D2H | Sync | No | 8.19 | 2.17 |
| v7_async_fence_optimized | D2H | Batch | Yes | 11.10 | 9.52 |
| v7_async_fence_optimized | D2H | Batch | No | 11.05 | 9.29 |
| v8_aggressive_batch | D2D | Sync | Yes | 178.95 | 1.46 |
| v8_aggressive_batch | D2D | Sync | No | 188.30 | 5.18 |
| v8_aggressive_batch | D2D | Batch | Yes | 208.53 | 41.24 |
| v8_aggressive_batch | D2D | Batch | No | 208.50 | 41.39 |
| v8_aggressive_batch | H2D | Sync | Yes | 10.23 | 4.13 |
| v8_aggressive_batch | H2D | Sync | No | 7.32 | 2.14 |
| v8_aggressive_batch | H2D | Batch | Yes | 10.27 | 8.23 |
| v8_aggressive_batch | H2D | Batch | No | 10.25 | 8.09 |
| v8_aggressive_batch | D2H | Sync | Yes | 11.04 | 4.37 |
| v8_aggressive_batch | D2H | Sync | No | 7.74 | 2.10 |
| v8_aggressive_batch | D2H | Batch | Yes | 11.10 | 9.52 |
| v8_aggressive_batch | D2H | Batch | No | 11.04 | 9.29 |
| v9_adaptive_coalescing_async | D2D | Sync | Yes | 188.70 | 6.20 |
| v9_adaptive_coalescing_async | D2D | Sync | No | 167.16 | 3.34 |
| v9_adaptive_coalescing_async | D2D | Batch | Yes | 208.54 | 41.17 |
| v9_adaptive_coalescing_async | D2D | Batch | No | 208.59 | 41.48 |
| v9_adaptive_coalescing_async | H2D | Sync | Yes | 10.23 | 4.13 |
| v9_adaptive_coalescing_async | H2D | Sync | No | 7.33 | 1.91 |
| v9_adaptive_coalescing_async | H2D | Batch | Yes | 10.28 | 8.23 |
| v9_adaptive_coalescing_async | H2D | Batch | No | 10.26 | 8.10 |
| v9_adaptive_coalescing_async | D2H | Sync | Yes | 11.04 | 4.37 |
| v9_adaptive_coalescing_async | D2H | Sync | No | 7.80 | 1.95 |
| v9_adaptive_coalescing_async | D2H | Batch | Yes | 11.09 | 9.52 |
| v9_adaptive_coalescing_async | D2H | Batch | No | 11.06 | 9.32 |
| Generic SYCL | D2D | Sync | Yes | 164.27 | 0.95 |
| Generic SYCL | D2D | Sync | No | 154.96 | 1.93 |
| Generic SYCL | D2D | Batch | Yes | 201.81 | 14.89 |
| Generic SYCL | D2D | Batch | No | 201.66 | 12.75 |
| Generic SYCL | H2D | Sync | Yes | 10.08 | 1.64 |
| Generic SYCL | H2D | Sync | No | 7.81 | 2.08 |
| Generic SYCL | H2D | Batch | Yes | 10.24 | 6.72 |
| Generic SYCL | H2D | Batch | No | 10.25 | 6.59 |
| Generic SYCL | D2H | Sync | Yes | 10.86 | 1.84 |
| Generic SYCL | D2H | Sync | No | 8.26 | 2.11 |
| Generic SYCL | D2H | Batch | Yes | 11.08 | 7.42 |
| Generic SYCL | D2H | Batch | No | 11.05 | 7.32 |
