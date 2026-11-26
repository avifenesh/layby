# Layby-Dwell release files

The Hugging Face repo `Avifenesh/layby-dwell` holds these files. They are not in git.

## v2.0 (`main`, tag `v2.0`)

Standalone `Layby-Dwell v2` (`33.8M` params, zero `laya` dependency, CPU-ready):

| file | bytes | sha256 | from |
|---|---:|---|---|
| `model.safetensors` | 135,033,124 | `afe8a2f187d964aee4c7d7cb427b0c5448bf00f8ea8ad126fe6865689b30b50b` | `rung4_32m_all_corpora` (seed 0 champion), fp32, 104 tensors, 33,755,701 params, no metadata |
| `config.json` | 2,762 | `3f6170cca7c4d232c3de8b7b09d2702d1552fccd047a8be1c5b99472a7bcba74` | sanitized v2 config (`model_version: 2`, `arch: v2_gated`, `mode: two_tower`, `max_len: 1536`, 15 edges, 32 numeric feature stats, per-kind temperatures, `layer_mix_weights`) |
| `encoder/config.json` | 1,239 | `5efc494125d8f38bb5c6ffe7f9fda14e97b270edb747eda336fc755d51288ed2` | `jhu-clsp/ettin-encoder-32m` (10x384 ModernBERT config, no duplicated base weights) |
| `tokenizer/tokenizer.json` | 2,132,967 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` | `jhu-clsp/ettin-encoder-32m` |
| `tokenizer/tokenizer_config.json` | 20,810 | `3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2` | `jhu-clsp/ettin-encoder-32m` |
| `tokenizer/special_tokens_map.json` | 694 | `ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049` | `jhu-clsp/ettin-encoder-32m` |
| `README.md` | | | `model/MODEL_CARD.md` |
| `LICENSE` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | Apache-2.0 (the repo `LICENSE`) |

Base encoder: `jhu-clsp/ettin-encoder-32m` at revision `1b8ba06455dd44f80fc9c1ca9e22806157a57379` (Apache-2.0).

## Archived v1.0 (branch `v1`, tag `v1.0`, commit `ddc3e908ff137758d3fbf0317e04810de439a917`)

Legacy 421M Laya/ModernBERT-large weights remain available via `Dwell("avifenesh/layby-dwell", revision="v1")` (requires `pip install -e ".[dwell,dwell-v1]"`):

| file | bytes | sha256 | from |
|---|---:|---|---|
| `model.safetensors` | 1,685,197,088 | `375da825845560bf1ad1e7ef03df02f8dd7f0f3076a0d1d93e1808093305cede` | the v6 checkpoint (best step), fp32, 206 tensors, no metadata |
| `config.json` | 1,396 | `27465edcb561869015818308a1a2b1f27bf7b96ecedbfd7fd2212e1aabdfc160` | base Laya's model config with max_len 1024, head_max_len 256, plus `layby_dwell` |
| `encoder/config.json` | 2,083 | `bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979` | base Laya (ModernBERT-large config) |
| `tokenizer/tokenizer.json` | 3,583,228 | `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30` | base Laya |
| `tokenizer/tokenizer_config.json` | 308 | `50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12` | base Laya |

Base: `convaiinnovations/laya` at revision `7b928d828b7b0e022f929d9bd2e44165aa270148` (Apache-2.0).
