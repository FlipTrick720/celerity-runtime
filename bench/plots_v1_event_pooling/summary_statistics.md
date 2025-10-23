# Benchmark Summary Statistics

| Backend | Operation | Mode | Pinned | Peak (GiB/s) | Median (GiB/s) | Min Latency (μs) |
| --- | --- | --- | --- | --- | --- | --- |
| LEVEL_ZERO | D2D | Sync | Yes | 188.98 | 6.64 | 35.79 |
| LEVEL_ZERO | D2D | Sync | No | 178.28 | 3.59 | 50.73 |
| LEVEL_ZERO | D2D | Batch | Yes | 208.32 | 41.20 | 5.25 |
| LEVEL_ZERO | D2D | Batch | No | 208.30 | 41.51 | 5.15 |
| LEVEL_ZERO | H2D | Sync | Yes | 10.22 | 4.12 | 34.81 |
| LEVEL_ZERO | H2D | Sync | No | 7.36 | 1.92 | 92.62 |
| LEVEL_ZERO | H2D | Batch | Yes | 10.27 | 8.22 | 4.27 |
| LEVEL_ZERO | H2D | Batch | No | 10.24 | 8.02 | 4.56 |
| LEVEL_ZERO | D2H | Sync | Yes | 10.53 | 4.26 | 34.02 |
| LEVEL_ZERO | D2H | Sync | No | 7.40 | 1.94 | 91.63 |
| LEVEL_ZERO | D2H | Batch | Yes | 10.58 | 9.15 | 3.44 |
| LEVEL_ZERO | D2H | Batch | No | 10.55 | 8.86 | 3.78 |
