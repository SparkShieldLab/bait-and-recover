# Qwen3-8B validation artifact

Base model: `Qwen/Qwen3-8B`

Recommended recipe:

```bash
GPU_ID=0 bash experiments/sidechannel_suite/run_paper_table3_qwen3_8b.sh
```

Validated 200-trial minimum refusal rates:

| KL budget | Refusal |
|---:|---:|
| <=0.01 | 93% |
| <=0.05 | 79% |
| <=0.10 | 60% |
| <=0.20 | 44% |
| <=1.0 | 43% |

Clean-behavior preservation KL: `0.0099`.


Included files provide structured summaries and the 200-trial Heretic journal used to compute the table above.
