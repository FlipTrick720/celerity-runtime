# Benchmark Summary Statistics

| Backend | Operation | Mode | Pinned | Peak (GiB/s) | Median (GiB/s) | Min Latency (μs) |
| --- | --- | --- | --- | --- | --- | --- |
| LEVEL_ZERO | D2D | Sync | Yes | 189.13 | 6.24 | 39.07 |
| LEVEL_ZERO | D2D | Sync | No | 188.82 | 4.79 | 32.55 |
| LEVEL_ZERO | D2D | Batch | Yes | 208.14 | 41.35 | 5.23 |
| LEVEL_ZERO | D2D | Batch | No | 208.28 | 41.57 | 5.18 |
| LEVEL_ZERO | H2D | Sync | Yes | 10.22 | 4.12 | 34.77 |
| LEVEL_ZERO | H2D | Sync | No | 7.82 | 2.13 | 80.94 |
| LEVEL_ZERO | H2D | Batch | Yes | 10.27 | 8.22 | 4.27 |
| LEVEL_ZERO | H2D | Batch | No | 10.24 | 8.00 | 4.57 |
| LEVEL_ZERO | D2H | Sync | Yes | 10.53 | 4.28 | 34.05 |
| LEVEL_ZERO | D2H | Sync | No | 7.95 | 2.16 | 78.43 |
| LEVEL_ZERO | D2H | Batch | Yes | 10.59 | 9.07 | 3.45 |
| LEVEL_ZERO | D2H | Batch | No | 10.55 | 8.85 | 3.87 |
