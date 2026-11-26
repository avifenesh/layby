"""Layby-Dwell v2: compact, zero-laya bidirectional survival encoder (ettin-32m / ettin-17m).

Predicts calibrated, hardware-independent idle-duration survival curves P(T > E2[j]) across
15 horizons (0.5s .. 1800s) directly from server-view session state JSON/dicts.

Architecture (v2_gated):
  - Compact ModernBERT / Ettin bidirectional text encoder over compact delta text
    (no Laya prompt wrapper, no external laya package dependency)
  - Learnable softmax LayerMix over top 4 encoder layers + [CLS] + AttentionPool
  - Periodic/Fourier numerical embeddings + Residual SwiGLU numeric tower over 32 server-view features
  - Gated residual bilinear+concat fusion: z_num + sigmoid(gate) * delta_head(fused)
  - Direct 16-bin right-censored discrete survival logits + per-kind temperature calibration
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

EDGES = [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 120, 300, 600, 1800]
K = len(EDGES) + 1
CLIP_HI = 7 * 86400.0
CLIP_LOG_HI = math.log1p(CLIP_HI)

NUM_COLS = [
    "arg_len", "cmd_len", "timeout", "ctx_log", "ctx_missing", "n_tools",
    "sess_idx", "sess_age_log", "since_prev_log",
    "prev_gap_log", "prev_same_kind_log", "ewma_same_kind_log", "n_same_kind",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "hist_tool_n_log", "hist_tool_mean", "hist_tool_q90", "hist_tool_fracfast",
    "hist_prog_n_log", "hist_prog_mean", "hist_prog_q90", "hist_prog_fracfast",
    "hist_ukind_n_log", "hist_ukind_mean", "hist_ukind_q90", "hist_ukind_fracfast",
    "background_flag", "is_tool_kind", "prev_is_tool",
]

QUANT = ("int8w", "fp8w")
PRECISIONS = ("bf16", "fp32") + QUANT


def is_v2_config(cfg: dict) -> bool:
    """Detect whether a config.json describes a Layby-Dwell v2 checkpoint."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("model_version") == 2:
        return True
    if "num_cols" in cfg and "num_mean" in cfg and "num_std" in cfg:
        return True
    d = cfg.get("layby_dwell")
    return isinstance(d, dict) and (d.get("model_version") == 2 or "num_cols" in d)


def _fval(x) -> float:
    if x is None:
        return math.nan
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _safe_log1p(x) -> float:
    v = _fval(x)
    if not math.isfinite(v) or v < 0.0:
        return math.nan
    return math.log1p(min(v, CLIP_HI))


def _clip_log(x) -> float:
    v = _safe_log1p(x)
    if not math.isfinite(v):
        return math.nan
    return min(max(v, 0.0), CLIP_LOG_HI)


def build_compact_delta_text(kind, tool, prog, model, args_text, asst_text, user_text) -> str:
    """Compact text representation matching Layby-Dwell v2 training exactly."""
    hdr = f"kind={kind}"
    if tool:
        hdr += f" tool={tool}"
    if prog:
        hdr += f" prog={prog}"
    if model:
        hdr += f" model={model}"
    parts = [hdr]
    if args_text:
        parts.append("TOOL_CALL: " + str(args_text))
    if asst_text:
        parts.append("ASSISTANT: " + str(asst_text))
    if user_text:
        parts.append("USER: " + str(user_text))
    return "\n".join(parts)


def state_to_v2_features(state) -> tuple[np.ndarray, str, str]:
    """Convert one server-view state (JSON string or dict) to (raw_num[32], delta_text, kind)."""
    if isinstance(state, str):
        try:
            d = json.loads(state)
        except ValueError:
            d = {}
    elif isinstance(state, dict):
        d = state
    else:
        d = {}

    raw_kind = str(d.get("kind") or "human")
    kind = "human" if raw_kind == "workflow" else raw_kind
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    th = d.get("tool_history") if isinstance(d.get("tool_history"), dict) else {}
    ph = meta.get("program_history") if isinstance(meta.get("program_history"), dict) else {}
    uh = meta.get("user_kind_history") if isinstance(meta.get("user_kind_history"), dict) else {}

    ctx_v = _fval(d.get("ctx_tokens"))
    ctx_missing = (not math.isfinite(ctx_v)) or (ctx_v <= 0.0)
    ctx_log = math.nan if ctx_missing else _safe_log1p(ctx_v)

    hour = _fval(d.get("hour"))
    if not math.isfinite(hour):
        hour = 0.0
    dow = _fval(d.get("dow"))
    if not math.isfinite(dow):
        dow = 0.0

    bg = meta.get("background")
    bg_flag = 1.0 if bool(bg) else 0.0
    prev_kind = str(meta.get("prev_kind") or "")

    row = np.array([
        _safe_log1p(d.get("arg_chars")),
        _safe_log1p(meta.get("cmd_chars")),
        _safe_log1p(meta.get("timeout_s")),
        ctx_log,
        1.0 if ctx_missing else 0.0,
        _safe_log1p(meta.get("n_tools")),
        _safe_log1p(d.get("turn")),
        _safe_log1p(d.get("session_age_s")),
        _safe_log1p(meta.get("since_prev_s")),
        _clip_log(meta.get("prev_gap_s")),
        _clip_log(meta.get("prev_same_kind_s")),
        _clip_log(meta.get("ewma_same_kind_s")),
        _safe_log1p(meta.get("n_same_kind")),
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * dow / 7.0),
        math.cos(2.0 * math.pi * dow / 7.0),
        _safe_log1p(th.get("n")),
        _clip_log(th.get("mean_s")),
        _clip_log(th.get("p90_s")),
        _fval(th.get("share_under_1s")),
        _safe_log1p(ph.get("n")),
        _clip_log(ph.get("mean_s")),
        _clip_log(ph.get("p90_s")),
        _fval(ph.get("share_under_1s")),
        _safe_log1p(uh.get("n")),
        _clip_log(uh.get("mean_s")),
        _clip_log(uh.get("p90_s")),
        _fval(uh.get("share_under_1s")),
        bg_flag,
        1.0 if kind == "tool" else 0.0,
        1.0 if prev_kind == "tool" else 0.0,
    ], dtype=np.float32)

    txt = build_compact_delta_text(
        kind=kind,
        tool=d.get("tool"),
        prog=d.get("program"),
        model=d.get("model"),
        args_text=d.get("tool_call") or "",
        asst_text=d.get("assistant") or "",
        user_text=d.get("user") or "",
    )
    return row, txt, kind


def normalize_numeric(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize (N, F) raw numeric matrix using train mean/std and produce explicit missing mask."""
    miss = ~np.isfinite(raw)
    vals = np.where(miss, 0.0, raw)
    normed = np.where(miss, 0.0, np.clip((vals - mean) / std, -10.0, 10.0)).astype(np.float32)
    return normed, miss.astype(np.float32)


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(h).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        w = torch.softmax(scores.float(), dim=-1).to(h.dtype).unsqueeze(-1)
        return (h * w).sum(dim=1)


class LayerMix(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.logits = nn.Parameter(torch.linspace(-0.5, 0.5, n_layers))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        top = hidden_states[-self.n_layers:]
        w = torch.softmax(self.logits.float(), dim=0)
        out = torch.zeros_like(top[0])
        for wi, hi in zip(w, top):
            out = out + wi.to(hi.dtype) * hi
        return out


class SwiGLUResBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.w_gate_val = nn.Linear(dim, inner * 2)
        self.w_out = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        g, v = self.w_gate_val(h).chunk(2, dim=-1)
        h = F.silu(g) * v
        return x + self.drop(self.w_out(h))


class NumericEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(self, num_vals: torch.Tensor, num_miss: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([num_vals, num_miss], dim=-1))


class PeriodicNumericEncoder(nn.Module):
    def __init__(self, n_features: int, n_freq: int = 8, hidden: int = 256, n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_freq) * 0.5)
        in_dim = n_features * (2 + 2 * n_freq)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([SwiGLUResBlock(hidden, mult=2, dropout=dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, num_vals: torch.Tensor, num_miss: torch.Tensor) -> torch.Tensor:
        obs = (1.0 - num_miss).unsqueeze(-1)
        angles = 2.0 * math.pi * num_vals.unsqueeze(-1) * self.freq.unsqueeze(0)
        sin_f = (torch.sin(angles) * obs).flatten(1)
        cos_f = (torch.cos(angles) * obs).flatten(1)
        x = torch.cat([num_vals, num_miss, sin_f, cos_f], dim=-1)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)


class LaybyDwellV2(nn.Module):
    def __init__(
        self,
        encoder_path: str | None = None,
        mode: str = "two_tower",
        arch: str = "v2_gated",
        n_num: int = len(NUM_COLS),
        num_hidden: int | None = None,
        head_hidden: int = 256,
        dropout: float = 0.1,
        aux_prefix: bool = False,
        attn_implementation: str = "sdpa",
        dtype=torch.float32,
        load_pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.arch = arch
        self.aux_prefix = aux_prefix
        if num_hidden is None:
            num_hidden = 128 if arch == "v1_plain" else 256

        txt_dim = 0
        if mode in ("two_tower", "text_only"):
            if encoder_path is None:
                raise ValueError("encoder_path is required for two_tower and text_only modes")
            if load_pretrained_encoder:
                self.encoder = AutoModel.from_pretrained(
                    encoder_path, dtype=dtype, attn_implementation=attn_implementation
                )
            else:
                enc_cfg = AutoConfig.from_pretrained(encoder_path)
                enc_cfg._attn_implementation = attn_implementation
                self.encoder = AutoModel.from_config(enc_cfg)
                if dtype is not None:
                    self.encoder = self.encoder.to(dtype=dtype)
            d_enc = self.encoder.config.hidden_size
            self.layer_mix = LayerMix(n_layers=min(4, self.encoder.config.num_hidden_layers)) if arch == "v2_gated" else None
            self.pool = AttentionPool(d_enc)
            txt_dim = d_enc * 2
        else:
            self.encoder = None
            self.layer_mix = None
            self.pool = None

        if mode in ("two_tower", "tabular_only"):
            if arch == "v2_gated":
                self.num_enc = PeriodicNumericEncoder(n_num, n_freq=8, hidden=num_hidden, n_blocks=2, dropout=dropout)
            else:
                self.num_enc = NumericEncoder(n_num, hidden=num_hidden, dropout=dropout)
        else:
            self.num_enc = None

        if arch == "v2_gated" and mode == "two_tower":
            self.num_head = nn.Linear(num_hidden, K)
            self.txt_to_num = nn.Sequential(
                nn.Linear(txt_dim, num_hidden),
                nn.LayerNorm(num_hidden),
                nn.SiLU(),
            )
            fuse_in = txt_dim + num_hidden * 2
            self.fuse_trunk = nn.Sequential(
                nn.Linear(fuse_in, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                SwiGLUResBlock(head_hidden, mult=2, dropout=dropout),
            )
            self.delta_head = nn.Linear(head_hidden, K)
            self.gate_head = nn.Linear(head_hidden, K)
            self.head = None
            pref_in = head_hidden
        else:
            in_dim = txt_dim + (num_hidden if self.num_enc is not None else 0)
            if arch == "v2_gated":
                self.head = nn.Sequential(
                    nn.Linear(in_dim, head_hidden),
                    nn.LayerNorm(head_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    SwiGLUResBlock(head_hidden, mult=2, dropout=dropout),
                    nn.Linear(head_hidden, K),
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(in_dim, head_hidden),
                    nn.LayerNorm(head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, K),
                )
            self.num_head = self.txt_to_num = self.fuse_trunk = self.delta_head = self.gate_head = None
            pref_in = in_dim

        self.prefix_head = nn.Linear(pref_in, 1) if aux_prefix else None

    def forward(self, input_ids=None, attention_mask=None, num_vals=None, num_miss=None):
        h_txt = None
        if self.encoder is not None:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(self.layer_mix is not None),
            )
            h = self.layer_mix(out.hidden_states) if self.layer_mix is not None else out.last_hidden_state
            cls_h = h[:, 0]
            att_h = self.pool(h, attention_mask)
            h_txt = torch.cat([cls_h, att_h], dim=-1)

        h_num = self.num_enc(num_vals, num_miss) if self.num_enc is not None else None

        if self.arch == "v2_gated" and self.mode == "two_tower":
            z_num = self.num_head(h_num)
            t_proj = self.txt_to_num(h_txt)
            inter = h_num * t_proj
            fused = self.fuse_trunk(torch.cat([h_txt, h_num, inter], dim=-1))
            gate = torch.sigmoid(self.gate_head(fused))
            logits = z_num + gate * self.delta_head(fused)
            pref = self.prefix_head(fused).squeeze(-1) if self.prefix_head is not None else None
            return logits, pref

        feats = [x for x in (h_txt, h_num) if x is not None]
        fused = torch.cat(feats, dim=-1)
        logits = self.head(fused)
        pref = self.prefix_head(fused).squeeze(-1) if self.prefix_head is not None else None
        return logits, pref


def logits_to_survival(logits: np.ndarray, kinds, temps: dict[str, float], n_horizons: int = len(EDGES)) -> np.ndarray:
    """Apply per-kind temperature and convert K=16 bin logits to (N, 15) monotonic survival P(T > E[j])."""
    t = np.array([temps.get(str(k), temps.get("_all", 1.0)) for k in kinds], dtype=np.float64)[:, None]
    z = logits.astype(np.float64) / np.maximum(t, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    S = 1.0 - np.cumsum(p, axis=1)[:, :n_horizons]
    return np.clip(S, 0.0, 1.0).astype(np.float32)


def _quantize(model, precision):
    from torchao.quantization import quantize_
    import torchao.quantization as q
    cfg = q.Int8WeightOnlyConfig() if precision == "int8w" else q.Float8WeightOnlyConfig()
    enc = model.encoder if hasattr(model, "encoder") and model.encoder is not None else model
    quantize_(enc, cfg, filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear))
    return model


def _resolve_encoder(base_dir: str, cfg: dict, encoder_path: str | None) -> str | None:
    if encoder_path:
        return encoder_path
    env_enc = os.environ.get("LAYBY_V2_ENCODER")
    if env_enc and os.path.isdir(env_enc):
        return env_enc
    local_enc = os.path.join(base_dir, "encoder")
    if os.path.isdir(local_enc):
        return local_enc
    args_enc = (cfg.get("args") or {}).get("encoder") or cfg.get("encoder")
    if args_enc and os.path.isdir(args_enc):
        return args_enc
    # Portable fallback if config recorded another user's home dir with the same relative path
    if args_enc and "/layby-data/" in args_enc:
        rel = args_enc.split("/layby-data/", 1)[1]
        for root in (os.path.expanduser("~/layby-data"), "/mnt/Data/layby-data"):
            cand = os.path.join(root, rel)
            if os.path.isdir(cand):
                return cand
    return args_enc


class DwellV2:
    """Zero-laya inference wrapper for Layby-Dwell v2 checkpoints (GPU BF16 or CPU FP32)."""

    model_version = 2

    def __init__(
        self,
        ckpt: str,
        calib: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        batch_size: int = 32,
        max_len: int | None = None,
        encoder_path: str | None = None,
        warmup: bool = True,
    ):
        if os.path.isdir(ckpt):
            base_dir = ckpt
            calib = calib or os.path.join(base_dir, "config.json")
            for name in ("best.safetensors", "model.safetensors"):
                cand = os.path.join(base_dir, name)
                if os.path.exists(cand):
                    ckpt = cand
                    break
        else:
            base_dir = os.path.dirname(os.path.abspath(ckpt))
            calib = calib or os.path.join(base_dir, "config.json")

        cfg = json.load(open(calib))
        d = cfg.get("layby_dwell", cfg)
        args = d.get("args") or {}

        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision or ("bf16" if self.dev.type == "cuda" else "fp32")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision is one of {PRECISIONS}")
        if self.precision in QUANT and self.dev.type != "cuda":
            raise ValueError("8-bit precisions need a GPU")

        self.cfg = d
        self.mode = args.get("mode", d.get("mode", "two_tower"))
        self.arch = args.get("arch", d.get("arch", "v2_gated"))
        self.max_len = int(max_len or args.get("max_len") or d.get("max_len") or 1536)
        self.horizons = np.asarray(d.get("edges", d.get("horizons_s", EDGES)), np.float32)
        self.temp = dict(d.get("temperature") or {"_all": 1.0})
        self.num_cols = list(d.get("num_cols", NUM_COLS))
        self.num_mean = np.asarray(d["num_mean"], np.float32)
        self.num_std = np.asarray(d["num_std"], np.float32)
        self.batch_size = int(batch_size)

        enc_path = None if self.mode == "tabular_only" else _resolve_encoder(base_dir, d, encoder_path)
        tok_dir = os.path.join(base_dir, "tokenizer")
        if self.mode != "tabular_only":
            self.tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else enc_path)
            self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        else:
            self.tok = None
            self.pad_id = 0

        aux_prefix = bool(args.get("aux_prefix_weight", 0.0) > 0 or d.get("aux_prefix", False))
        m = LaybyDwellV2(
            encoder_path=enc_path,
            mode=self.mode,
            arch=self.arch,
            n_num=len(self.num_cols),
            aux_prefix=aux_prefix,
            attn_implementation="sdpa",
            dtype=torch.float32,
            load_pretrained_encoder=False,
        )
        m.load_state_dict(load_file(ckpt), strict=True)
        m.eval()
        self.model = m.to(self.dev)
        self._amp = torch.bfloat16 if (self.precision == "bf16" and self.dev.type == "cuda") else None
        if self.precision in QUANT:
            self.model = _quantize(self.model.to(torch.bfloat16), self.precision)
        if warmup and self.dev.type == "cuda":
            self.warmup()

    @torch.no_grad()
    def warmup(self, lengths=(64, 512), batches=(1, 4)):
        for n in batches:
            nv = torch.zeros((n, len(self.num_cols)), dtype=torch.float32, device=self.dev)
            nm = torch.ones((n, len(self.num_cols)), dtype=torch.float32, device=self.dev)
            if self.mode == "tabular_only":
                self.model(num_vals=nv, num_miss=nm)
                continue
            for L in lengths:
                L = min(L, self.max_len)
                ids = torch.full((n, L), self.pad_id, dtype=torch.long, device=self.dev)
                att = torch.ones((n, L), dtype=torch.long, device=self.dev)
                with torch.autocast(self.dev.type, dtype=self._amp, enabled=(self._amp is not None)):
                    self.model(input_ids=ids, attention_mask=att, num_vals=nv, num_miss=nm)
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    @torch.no_grad()
    def logits_from_features(self, texts: list[str], raw_num: np.ndarray) -> np.ndarray:
        """Score pre-extracted (texts, raw_num) directly (useful for exact parity checks on parquet splits)."""
        n = len(raw_num)
        if n == 0:
            return np.zeros((0, K), np.float32)
        nv_all, nm_all = normalize_numeric(raw_num, self.num_mean, self.num_std)
        Z = np.zeros((n, K), np.float32)
        if self.mode == "tabular_only":
            for s in range(0, n, self.batch_size):
                e = min(n, s + self.batch_size)
                nv = torch.from_numpy(nv_all[s:e]).to(self.dev)
                nm = torch.from_numpy(nm_all[s:e]).to(self.dev)
                with torch.autocast(self.dev.type, dtype=self._amp, enabled=(self._amp is not None)):
                    lg, _ = self.model(num_vals=nv, num_miss=nm)
                Z[s:e] = lg.float().cpu().numpy()
            return Z

        enc = self.tok(
            texts,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        order = sorted(range(n), key=lambda i: len(enc[i]))
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            max_l = max(len(enc[i]) for i in idx)
            L = max(16, ((max_l + 15) // 16) * 16)
            bs = len(idx)
            ids = torch.full((bs, L), self.pad_id, dtype=torch.long)
            att = torch.zeros((bs, L), dtype=torch.long)
            for r, i in enumerate(idx):
                seq = enc[i]
                ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                att[r, :len(seq)] = 1
            ids = ids.to(self.dev, non_blocking=True)
            att = att.to(self.dev, non_blocking=True)
            nv = torch.from_numpy(nv_all[idx]).to(self.dev, non_blocking=True)
            nm = torch.from_numpy(nm_all[idx]).to(self.dev, non_blocking=True)
            with torch.autocast(self.dev.type, dtype=self._amp, enabled=(self._amp is not None)):
                lg, _ = self.model(input_ids=ids, attention_mask=att, num_vals=nv, num_miss=nm)
            Z[idx] = lg.float().cpu().numpy()
        return Z

    def logits_and_survival(self, states, kinds=None, temp_override: dict[str, float] | None = None):
        """Convert server-view states (JSON strings or dicts) to (logits[N, 16], survival[N, 15])."""
        if not states:
            return np.zeros((0, K), np.float32), np.zeros((0, len(self.horizons)), np.float32)
        rows = [state_to_v2_features(s) for s in states]
        raw_num = np.stack([r[0] for r in rows], axis=0)
        texts = [r[1] for r in rows]
        inferred_kinds = [r[2] for r in rows]
        use_kinds = inferred_kinds if kinds is None else list(kinds)
        Z = self.logits_from_features(texts, raw_num)
        temps = temp_override if temp_override is not None else self.temp
        S = logits_to_survival(Z, use_kinds, temps, len(self.horizons))
        return Z, S

    def survival(self, states, kinds=None) -> np.ndarray:
        _, S = self.logits_and_survival(states, kinds)
        return S

    def score(self, states, kinds=None) -> np.ndarray:
        return self.survival(states, kinds)

    def median_return(self, S: np.ndarray) -> np.ndarray:
        below = S <= 0.5
        return np.where(below.any(1), self.horizons[np.argmax(below, 1)], np.inf)
