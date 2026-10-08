"""Laya scorer for the sidecar: server-view states -> P(T > E2[j]) under the checkpoint's per-kind
temperature (the same computation as the offline scorer of the training repo, kept in memory and batched)."""
import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence


class Scorer:
    def __init__(self, ckpt, calib, device=None, max_len=1024):
        cal = json.load(open(calib))
        cal = cal.get("layby_dwell", cal)     # the Layby-Dwell release config.json carries it under "layby_dwell"
        self.H = cal.get("H", cal.get("horizons_s"))
        self.levels, self.temp = cal["levels"], cal["temperature"]
        self.K = len(self.levels)
        self.q = {"t": "score",
                  "ins": "The model just finished this turn of an LLM session and its KV cache is now idle. "
                         "How long until the session's next request arrives at the server?",
                  "crit": self.levels}
        model_dir = os.environ.get("LAYA_DIR") or snapshot_download("convaiinnovations/laya")
        _fix_tokenizer_config(model_dir)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
        cfg["gradient_checkpointing"] = False; cfg["max_len"] = max_len; cfg["head_max_len"] = 256
        self.cfg = cfg
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
        self.model.load_state_dict(load_file(ckpt), strict=True)
        self.model.to(self.dev).eval()

    @torch.no_grad()
    def score(self, states, kinds):
        """states: JSON strings (server view); kinds: "tool" or "human". Returns (n, 15) survival rows;
        a state the sequence builder cannot fit gets NaN (the adapter treats NaN as 0.5)."""
        items = [build_sequence(self.tok, s, self.q, self.cfg["max_len"], self.cfg["head_max_len"]) for s in states]
        keep = [i for i, (s, m) in enumerate(items) if len(m) == self.K]
        Z = np.full((len(states), self.K), np.nan, np.float32)
        if keep:
            n = len(keep); L = max(len(items[i][0]) for i in keep)
            ids = torch.full((n, L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((n, L), dtype=torch.long); mpos = torch.zeros((n, self.K), dtype=torch.long)
            for r, i in enumerate(keep):
                s, m = items[i]; ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1; mpos[r] = torch.tensor(m)
            args = (ids.to(self.dev), att.to(self.dev), mpos.to(self.dev),
                    torch.ones((n, self.K), dtype=torch.bool, device=self.dev),
                    torch.full((n,), QTYPES["score"], dtype=torch.long, device=self.dev))
            if self.dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = self.model(*args)
            else:
                lg, _ = self.model(*args)
            Z[keep] = lg.float().cpu().numpy()
        t = np.array([self.temp.get(k, self.temp["_all"]) for k in kinds], np.float32)[:, None]
        z = Z / t; z = z - np.nanmax(z, 1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        return np.clip(1.0 - np.cumsum(P, 1)[:, :len(self.H)], 0, 1)
