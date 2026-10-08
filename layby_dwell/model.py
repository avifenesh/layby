"""Layby-Dwell: server-view session state -> P(T > h) at 15 horizons, T = seconds until the session's
next request.

One forward pass per state. States are batched, sorted by token length and padded only to the
longest state of their batch (rounded up to a multiple of 64), so a mix of short and long states
does not pay for the longest one everywhere. The output is calibrated by kind (tool, human) with the
temperatures fitted after training.

Precision: "bf16" (default on GPU: the precision the model was trained and evaluated in) or "fp32"
(default on CPU). 8-bit weights on GPU (torchao, the encoder's linear layers; the scoring head stays bf16):
"int8w" or "fp8w" (fp8w needs an FP8 GPU: Ada, Hopper or Blackwell). They halve the encoder's weight memory, are
not faster (the forward pass is bound by attention over padded sequences, not by the linear layers), and pick
the same cost-rule option as bf16 for 99.75% (int8w) and 99.54% (fp8w) of public states. model/QUANT.md has the
measurements; 8-bit activations (int8 and fp8 dynamic) failed and are not offered.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence

REPO = "avifenesh/layby-dwell"
QUANT = ("int8w", "fp8w")
PRECISIONS = ("bf16", "fp32") + QUANT


def _quantize(model, precision):
    """torchao 8-bit for the encoder's linear layers (the scoring head stays in bf16)."""
    from torchao.quantization import quantize_
    import torchao.quantization as q
    cfg = q.Int8WeightOnlyConfig() if precision == "int8w" else q.Float8WeightOnlyConfig()
    enc = model.encoder if hasattr(model, "encoder") else model
    quantize_(enc, cfg, filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear))
    return model


class Dwell:
    def __init__(self, path=None, device=None, precision=None, batch_size=32):
        """path: a local copy of the model repo (config.json, model.safetensors, encoder/, tokenizer/);
        None downloads REPO from the Hugging Face Hub."""
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision or ("bf16" if self.dev.type == "cuda" else "fp32")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        if self.precision in QUANT and self.dev.type != "cuda":
            raise ValueError("8-bit precisions need a GPU")
        if path is None or not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path or REPO)
        cfg = json.load(open(os.path.join(path, "config.json")))
        d = cfg["layby_dwell"]
        self.horizons = np.asarray(d["horizons_s"], np.float32)
        self.temp = d["temperature"]
        self.K = len(d["levels"])
        self.q = dict(d["question"], crit=d["levels"])
        self.cfg = cfg
        _fix_tokenizer_config(path)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        self.batch_size = batch_size
        m = build_model(cfg, encoder_dir=os.path.join(path, "encoder"))
        m.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
        m.eval()
        self.model = m.to(self.dev)
        self._amp = torch.bfloat16 if self.precision == "bf16" else None
        if self.precision in QUANT:
            self.model = _quantize(self.model.to(torch.bfloat16), self.precision)

    def encode(self, states):
        """Token ids and the positions of the 15 + 1 answer markers per state; None where a state
        cannot be fitted into max_len."""
        out = []
        for s in states:
            s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, default=str)
            seq, mk = build_sequence(self.tok, s, self.q, self.cfg["max_len"], self.cfg["head_max_len"])
            out.append((seq, mk) if len(mk) == self.K else None)
        return out

    @torch.no_grad()
    def logits(self, items):
        Z = np.full((len(items), self.K), np.nan, np.float32)
        keep = sorted((i for i, x in enumerate(items) if x is not None), key=lambda i: len(items[i][0]))
        for b in range(0, len(keep), self.batch_size):
            idx = keep[b:b + self.batch_size]
            L = -(-max(len(items[i][0]) for i in idx) // 64) * 64
            n = len(idx)
            ids = torch.full((n, L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((n, L), dtype=torch.long)
            mpos = torch.zeros((n, self.K), dtype=torch.long)
            for r, i in enumerate(idx):
                seq, mk = items[i]
                ids[r, :len(seq)] = torch.tensor(seq); att[r, :len(seq)] = 1; mpos[r] = torch.tensor(mk)
            args = (ids.to(self.dev), att.to(self.dev), mpos.to(self.dev),
                    torch.ones((n, self.K), dtype=torch.bool, device=self.dev),
                    torch.full((n,), QTYPES["score"], dtype=torch.long, device=self.dev))
            if self._amp is not None:
                with torch.autocast(self.dev.type, dtype=self._amp):
                    lg, _ = self.model(*args)
            else:
                lg, _ = self.model(*args)
            Z[idx] = lg.float().cpu().numpy()
        return Z

    def survival(self, states, kinds):
        """states: server-view states (JSON strings or dicts, see session.Tracker); kinds: "tool" or
        "human" per state. Returns an (n, 15) array of P(T > horizons[j]). A state that cannot be
        fitted gives a row of NaN; treat it as unknown (the engine adapters fall back to LRU)."""
        Z = self.logits(self.encode(states))
        t = np.array([self.temp.get(k, self.temp["_all"]) for k in kinds], np.float32)[:, None]
        z = Z / t
        ok = ~np.isnan(z).all(1)
        S = np.full((len(states), len(self.horizons)), np.nan, np.float32)
        if ok.any():
            z = z[ok] - z[ok].max(1, keepdims=True)
            P = np.exp(z); P /= P.sum(1, keepdims=True)
            S[ok] = np.clip(1.0 - np.cumsum(P, 1)[:, :len(self.horizons)], 0, 1)
        return S

    def median_return(self, S):
        """Median predicted return in seconds per row (the first horizon where P(T > h) <= 0.5;
        inf when the session is more likely than not to stay away past the last horizon)."""
        below = S <= 0.5
        return np.where(below.any(1), self.horizons[np.argmax(below, 1)], np.inf)
