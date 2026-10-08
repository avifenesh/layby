# Layby-Dwell in 8-bit

8-bit weights work and keep the rule's decisions. They do not make scoring faster. Keep bf16 as the default; use
int8 weights when GPU memory next to the engine is tight.

## Setup

- GPU: one RTX 4090 (24 GB), torch 2.13.0+cu130, torchao 0.18.0 (Vast instance 54695545, 2026-10-07).
- States: 4,445 public-pool states (SWE-chat 2,044, WildChat 1,746, TraceLab 655), the same states the
  ReturnBench pools are scored from. Median 740 tokens, p90 1,000.
- Reference: bf16, the precision every evaluation used.
- 8-bit applies to the encoder's linear layers through torchao. The scoring head stays in bf16.
- Decisions: the cost rule (`sim.rule.decide`) on each state's curve at 16k and 32k session tokens, against one
  fixed loaded engine state (`layby_dwell/quant_report.py`). bf16 picks none 2,793 times, park 3,560 and drop
  2,537, so agreement is measured on a real mix. At 4k tokens every curve gives drop, so 4k is left out.
- Timings are medians of 3 runs. Throughput covers the first 1,024 states, tokenizing included.

## Results

| precision | max abs dP | p99 abs dP | mean abs dP | same decision | one state | states/s at batch 1 / 8 / 32 / 128 |
|---|---|---|---|---|---|---|
| bf16 (reference) | 0 | 0 | 0 | 100% | 23.0 ms | 43.8 / 93.1 / 83.6 / 82.7 |
| fp32 | 0.0135 | 0.0036 | 0.0004 | 99.87% | 27.7 ms | 34.2 / 40.2 / 40.4 / out of memory |
| int8 weights (int8w) | 0.0308 | 0.0089 | 0.0010 | 99.75% | 26.5 ms | 38.2 / 90.3 / 81.2 / 79.6 |
| fp8 weights (fp8w) | 0.0819 | 0.0171 | 0.0019 | 99.54% | 29.1 ms | 34.6 / 95.2 / 88.6 / 87.3 |
| int8 weights and activations | 0.0702 | 0.0195 | 0.0021 | 99.43% | 167.0 ms | 6.1 / 31.6 / 37.2 / 28.9 |
| fp8 weights and activations | 0.1093 | 0.0242 | 0.0026 | 99.40% | 65.2 ms | 15.3 / 82.5 / 67.1 / 65.1 |

fp8 with fp8 activations returned NaN for 91 of the 4,445 states (2%). Those states are left out of its
columns.

Where the time goes (`layby_dwell/quant_split.py`): tokenizing runs at about 620 to 640 states/s on the CPU in
every precision. The forward pass runs at 95 to 109 states/s in bf16, int8w and fp8w, and it gets slower as the
batch grows. Attention over padded sequences bounds it, not the linear layers, so 8-bit weights cannot speed it up.

## Gate

The model card's gate is max abs dP <= 0.01. No 8-bit variant meets it, and neither does fp32: fp32 differs from
bf16 by up to 0.0135 on these 4,445 states. The earlier 0.0084 came from 256 states. So 0.01 sits below the
model's own bf16 to fp32 spread. The fallback gate is that the rule picks the same option for at least 99.5% of
states.

| precision | ships | why |
|---|---|---|
| int8w | yes | 99.75% same decisions, no NaN, same speed as bf16, half the encoder weight memory |
| fp8w | yes, on FP8 GPUs | 99.54% same decisions, no NaN, same speed as bf16 |
| int8 dynamic | no | 99.43%, and 7x slower at one state |
| fp8 dynamic | no | NaN on 2% of states |

`Dwell(precision="int8w")` and `precision="fp8w"` are the shipped options (`layby_dwell/model.py`). The test is
`tests/test_dwell_precision.py`. Its GPU part needs `LAYBY_DWELL_DIR` and `LAYBY_STATES`.

## Recommendation

- Serve in bf16. It is the fastest, and it is what every number in the paper used.
- Use int8w when the scorer shares a GPU with the engine and memory matters.
- Faster scoring comes from shorter or less padded inputs and attention kernels, not from 8-bit weights.

Raw rows and logs (out.npz, report.txt, split.txt and the run logs) are kept with the author and available on request.
