# Paper Result Bundles (`data/`)

Content-free JSON and JSONL summary files used by `figs.py` to regenerate every table and figure in the Layby paper (Apache-2.0):

- `v2_results.json`: real-engine vLLM 0.34.0 / nightly, SGLang 0.5.23, cross-hardware single-GPU, slow-disk ladder, fast-NVMe decomposition, and full `eval_uhold` (`n=184,148`) model ablation summaries.
- `returnbench_v2_s0.json`: 33-cell ReturnBench evaluation for published 33.8M Layby-Dwell v2 (`rung4_32m_all_corpora`, seed 0; 100% curve coverage on TraceLab and WildChat, 99.2% on SWE-chat).
- `ablation_rows.jsonl`: 33-cell ReturnBench action-option ablations (Table 10; uses archived 421M v1 `surv_v6` curves for `layby` rows).
- `baselines_rows.jsonl`: 33-cell ReturnBench GPU-pinning baselines (Table 11: Continuum, Choi-Joshi, Keep-Alive).

Regenerate all paper tables and figures from the repository root (`pip install -e ".[figs]"`):

```bash
python3 figs.py
```
