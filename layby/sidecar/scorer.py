"""Sidecar scorer: server-view states -> P(T > E2[j]) under per-kind temperature calibration.

Auto-detects Layby-Dwell v2 (compact Ettin 33.8M / 18.5M, zero laya dependency, GPU BF16 or CPU FP32)
and legacy Layby-Dwell v1 (421M Laya, lazy-imported only for v1 checkpoints).
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from layby_dwell.v2 import DwellV2, is_v2_config


class Scorer:
    def __init__(self, ckpt, calib, device=None, max_len=None, precision=None, encoder_path=None):
        cfg = json.load(open(calib))
        if is_v2_config(cfg):
            self._v2 = DwellV2(
                ckpt=ckpt,
                calib=calib,
                device=device,
                precision=precision,
                max_len=max_len,
                encoder_path=encoder_path,
                warmup=True,
            )
            self.model_version = 2
            self.dev = self._v2.dev
            self.H = self._v2.horizons.tolist()
            self.temp = self._v2.temp
            return

        # Legacy v1 (421M Laya) fallback: lazy import so v2 deployments require zero laya dependency
        self._v2 = None
        self.model_version = 1
        from huggingface_hub import snapshot_download
        from laya.agent import _fix_tokenizer_config
        from laya.common import QTYPES, build_model, build_sequence
        self._QTYPES, self._build_sequence = QTYPES, build_sequence

        cal = cfg.get("layby_dwell", cfg)
        self.H = cal.get("H", cal.get("horizons_s"))
        self.levels, self.temp = cal["levels"], dict(cal["temperature"])
        self.K = len(self.levels)
        self.q = {"t": "score",
                  "ins": "The model just finished this turn of an LLM session and its KV cache is now idle. "
                         "How long until the session's next request arrives at the server?",
                  "crit": self.levels}
        model_dir = os.environ.get("LAYA_DIR") or snapshot_download("convaiinnovations/laya")
        _fix_tokenizer_config(model_dir)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        lcfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
        lcfg["gradient_checkpointing"] = False
        lcfg["max_len"] = int(max_len or 1024)
        lcfg["head_max_len"] = 256
        self.cfg = lcfg
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(lcfg, encoder_dir=os.path.join(model_dir, "encoder"))
        self.model.load_state_dict(load_file(ckpt), strict=True)
        self.model.to(self.dev).eval()
        if self.dev.type == "cuda":
            self.warmup()

    @torch.no_grad()
    def warmup(self, lengths=(64, None), batches=(1, 8)):
        """One forward pass per (batch, length) before serving so CUDA kernels load before replay starts."""
        if self._v2 is not None:
            self._v2.warmup()
            return
        L_max = self.cfg["max_len"]
        for n in batches:
            for L in lengths:
                L = L or L_max
                args = (torch.full((n, L), self.tok.pad_token_id, dtype=torch.long, device=self.dev),
                        torch.ones((n, L), dtype=torch.long, device=self.dev),
                        torch.arange(self.K, device=self.dev).repeat(n, 1) % L,
                        torch.ones((n, self.K), dtype=torch.bool, device=self.dev),
                        torch.full((n,), self._QTYPES["score"], dtype=torch.long, device=self.dev))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.model(*args)
        torch.cuda.synchronize(self.dev)

    @torch.no_grad()
    def score_with_logits(self, states, kinds, temp_override=None):
        """Return (logits[n, K], survival[n, 15])."""
        if self._v2 is not None:
            return self._v2.logits_and_survival(states, kinds, temp_override=temp_override)
        items = [self._build_sequence(self.tok, s, self.q, self.cfg["max_len"], self.cfg["head_max_len"]) for s in states]
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
                    torch.full((n,), self._QTYPES["score"], dtype=torch.long, device=self.dev))
            if self.dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = self.model(*args)
            else:
                lg, _ = self.model(*args)
            Z[keep] = lg.float().cpu().numpy()
        temps = temp_override if temp_override is not None else self.temp
        t = np.array([temps.get(k, temps["_all"]) for k in kinds], np.float32)[:, None]
        z = Z / t; z = z - np.nanmax(z, 1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        S = np.clip(1.0 - np.cumsum(P, 1)[:, :len(self.H)], 0, 1)
        return Z, S

    @torch.no_grad()
    def score(self, states, kinds, temp_override=None):
        """states: JSON strings (server view); kinds: "tool" or "human". Returns (n, 15) survival rows."""
        _, S = self.score_with_logits(states, kinds, temp_override=temp_override)
        return S
