# Benchmark Summary Statistics

| Backend | Operation | Mode | Pinned | Peak (GiB/s) | Median (GiB/s) | Min Latency (μs) |
| --- | --- | --- | --- | --- | --- | --- |
| LEVEL_ZERO | D2D | Sync | Yes | 181.86 | 2.05 | 50.27 |
| LEVEL_ZERO | D2D | Sync | No | 189.33 | 6.77 | 32.65 |
| LEVEL_ZERO | D2D | Batch | Yes | 208.35 | 41.07 | 5.29 |
| LEVEL_ZERO | D2D | Batch | No | 208.27 | 41.56 | 5.25 |
| LEVEL_ZERO | H2D | Sync | Yes | 10.22 | 4.12 | 34.86 |
| LEVEL_ZERO | H2D | Sync | No | 7.30 | 2.05 | 89.22 |
| LEVEL_ZERO | H2D | Batch | Yes | 10.27 | 8.18 | 4.32 |
| LEVEL_ZERO | H2D | Batch | No | 10.26 | 8.03 | 4.59 |
| LEVEL_ZERO | D2H | Sync | Yes | 10.53 | 4.28 | 34.12 |
| LEVEL_ZERO | D2H | Sync | No | 7.25 | 2.09 | 86.68 |
| LEVEL_ZERO | D2H | Batch | Yes | 10.58 | 9.13 | 3.51 |
| LEVEL_ZERO | D2H | Batch | No | 10.56 | 8.86 | 3.72 |
