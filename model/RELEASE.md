# Layby-Dwell release files

The Hugging Face repo `avifenesh/layby-dwell` holds these files. They are not in git.

| file | bytes | sha256 | from |
|---|---|---|---|
| `model.safetensors` | 1,685,197,088 | `375da825845560bf1ad1e7ef03df02f8dd7f0f3076a0d1d93e1808093305cede` | the v6 checkpoint (best step), fp32, 206 tensors, no metadata |
| `config.json` | 1,396 | `27465edcb561869015818308a1a2b1f27bf7b96ecedbfd7fd2212e1aabdfc160` | base Laya's model config with max_len 1024, head_max_len 256, plus `layby_dwell`: horizons, levels, per-kind temperatures, the question text |
| `encoder/config.json` | 2,083 | `bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979` | base Laya (ModernBERT-large config) |
| `tokenizer/tokenizer.json` | 3,583,228 | `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30` | base Laya |
| `tokenizer/tokenizer_config.json` | 308 | `50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12` | base Laya |
| `README.md` | | | `model/MODEL_CARD.md` |
| `LICENSE` | | | Apache-2.0 (the repo LICENSE) |

Base: `convaiinnovations/laya` at revision `7b928d828b7b0e022f929d9bd2e44165aa270148` (Apache-2.0).

Check before upload: `python -m layby_dwell.bench STATES.parquet --model DIR` on public-pool states
reproduces the scores the paper used (largest difference 0.0084 on 256 states, fp32 CPU against the
bf16 GPU scores).

Tested from a clean environment on 2026-10-10: a fresh Python 3.12 venv with `pip install -e ".[dwell]"` (laya 0.4.2,
torch 2.14.1, transformers 5.19.0) ran the model card's usage snippet against the Hub repo on CPU: the model loaded in
36 s and returned 15 non-increasing probabilities for a tool-call state. `tests/test_dwell_precision.py` passed (the
GPU precision check skipped without a GPU).
