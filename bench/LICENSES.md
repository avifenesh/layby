# Sources and licenses

Each pool is derived from one public dataset and carries that dataset's license. The pools hold no text:
only session and turn indices, start times, token counts, idle gaps, the turn kind (human, tool,
workflow), tool names, the first program of a shell command, a pool-local user index and model survival
curves. Changes from the source data: sessions sampled, idle periods extracted as gaps, contexts scaled
under 32,768 tokens, user ids replaced by pool-local indices, MCP tool names merged to `mcp`, rare
program names merged to `other`. See `build/` for the exact steps.

| Pool | Source | Version used | License | Redistribution of derived traces |
|---|---|---|---|---|
| `swechat` | SWE-chat, SALT-NLP (Baumann et al., arXiv 2604.20779). https://huggingface.co/datasets/SALT-NLP/SWE-chat | revision `f66cca95b14caaa4177f7ed5eaa424608dadcffa` | ODC-BY 1.0 | Allowed with attribution. The dataset is gated on Hugging Face with an automatic click-through; the card lists no terms beyond ODC-BY. |
| `tracelab_claude` | TraceLab public coding-agent trace, SyFI Lab, University of Washington. https://github.com/uw-syfi/TraceLab, https://tracelab.cs.washington.edu | release `v0.0.2`, `syfi_coding_trace.duckdb` | CC BY 4.0 | Allowed with attribution and a note of changes. The license file asks users not to try to re-identify contributors; the pool keeps no TraceLab ids. |
| `wildchat` | WildChat-4.8M, Allen Institute for AI (Zhao et al., arXiv 2405.01470). https://huggingface.co/datasets/allenai/WildChat-4.8M | revision `c827c6df8fcf008219ffaffa4d1dd77491099367` | ODC-BY 1.0 | Allowed with attribution. Not gated. |
| `copilot` | GitHub Copilot Coding Agent Traces 2026, Azure Public Dataset (Liu et al., arXiv 2608.00101). https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md | release `ghcp-coding-agent-2026` | CC BY 4.0 | Allowed with attribution and a note of changes. The source has no text and no user ids. |

All four allow commercial use. No pool was dropped.

## Attribution

- SWE-chat: Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, Sanmi Koyejo.
  "SWE-chat: Real-World AI Coding Sessions in the Wild." arXiv 2604.20779, 2026. ODC-BY 1.0,
  https://opendatacommons.org/licenses/by/1-0/.
- TraceLab: TraceLab (SyFI Lab, University of Washington), https://tracelab.cs.washington.edu.
  CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
- WildChat: Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, Yuntian Deng.
  "WildChat: 1M ChatGPT Interaction Logs in the Wild." ICLR 2024, arXiv 2405.01470. WildChat-4.8M,
  ODC-BY 1.0.
- Copilot: Banruo Liu, Haoran Qiu, Íñigo Goiri, Rodrigo Fonseca, Ricardo Bianchini, Esha Choukse.
  "Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale."
  arXiv 2608.00101, 2026. CC BY 4.0.

## Removal requests

SWE-chat and WildChat take deletion requests from people whose data they hold. The pools keep no
source session or conversation ids, so a removal upstream cannot be matched here by id. Rebuilding with
`build/build_all.sh` against a later upstream revision picks up removals; the pinned revision and the
reference numbers change with it.
