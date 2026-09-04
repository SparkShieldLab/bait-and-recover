# Gemma-3-12B-it validation artifact

Base model: `google/gemma-3-12b-it`

Recommended recipe:

```bash
GPU_ID=1 bash experiments/sidechannel_suite/run_paper_table3_gemma3_12b.sh
```

Validated 200-trial minimum refusal rates:

| KL budget | Refusal |
|---:|---:|
| <=0.01 | 83% |
| <=0.05 | 83% |
| <=0.10 | 83% |
| <=0.20 | 83% |
| <=1.0 | 79% |

Clean-behavior preservation KL: `0.0850`.


Included files provide structured summaries and the 200-trial Heretic journal used to compute the table above.
