"""Looped (weight-shared, depth-recurrent) decoder LM, Ouro-compatible.

The module tree mirrors HF ``OuroForCausalLM`` so Ouro checkpoints load with
``load_state_dict`` directly. Engines in this package do not call
``LoopedLM.forward``; they drive the per-layer primitives below so the dense
baseline and FlashLoop share exactly the same math.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoopedConfig:
    vocab_size: int = 49152
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    total_ut_steps: int = 4  # number of loops R
    sandwich_norm: bool = True  # Ouro uses extra post-attn / post-mlp norms
    norm_between_loops: bool = True  # Ouro applies the final norm after every loop
    tie_word_embeddings: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_hf(cls, path: str | Path) -> "LoopedConfig":
        raw = json.loads((Path(path) / "config.json").read_text())
        names = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kw = {k: v for k, v in raw.items() if k in names}
        kw.setdefault("head_dim", raw["hidden_size"] // raw["num_attention_heads"])
        return cls(**kw, extra={k: v for k, v in raw.items() if k not in names})

    @classmethod
    def tiny(cls, **kw) -> "LoopedConfig":
        base = dict(vocab_size=256, hidden_size=128, intermediate_size=352, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=32, total_ut_steps=4)
        base.update(kw)
        return cls(**base)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dt)


class MLP(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        h, d = cfg.hidden_size, cfg.head_dim
        self.n_heads, self.n_kv, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, d
        self.q_proj = nn.Linear(h, self.n_heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * d, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * d, bias=False)
        self.o_proj = nn.Linear(self.n_heads * d, h, bias=False)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.sandwich = cfg.sandwich_norm
        if cfg.sandwich_norm:
            self.input_layernorm_2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.post_attention_layernorm_2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    # --- primitives used by the engines -------------------------------------------------
    def qkv(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """h: [B, T, D] residual stream -> q [B,H,T,d], k/v [B,Hkv,T,d] (post-RoPE)."""
        a = self.self_attn
        x = self.input_layernorm(h)
        B, T, _ = x.shape
        q = a.q_proj(x).view(B, T, a.n_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(x).view(B, T, a.n_kv, a.head_dim).transpose(1, 2)
        v = a.v_proj(x).view(B, T, a.n_kv, a.head_dim).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def finish(self, h: torch.Tensor, attn_heads: torch.Tensor) -> torch.Tensor:
        """attn_heads: [B,H,T,d] per-head attention output (pre o_proj)."""
        B, H, T, d = attn_heads.shape
        o = self.self_attn.o_proj(attn_heads.transpose(1, 2).reshape(B, T, H * d))
        if self.sandwich:
            o = self.input_layernorm_2(o)
        h = h + o
        m = self.mlp(self.post_attention_layernorm(h))
        if self.sandwich:
            m = self.post_attention_layernorm_2(m)
        return h + m


class LoopedLM(nn.Module):
    """Module layout == HF Ouro: ``model.{embed_tokens,layers,norm,early_exit_gate}``, ``lm_head``."""

    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.model.early_exit_gate = nn.Linear(cfg.hidden_size, 1)  # loaded, unused (fixed R loops)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    @property
    def layers(self):
        return self.model.layers

    @property
    def n_loops(self) -> int:
        return self.cfg.total_ut_steps

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(ids)

    def end_of_loop(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.norm(h) if self.cfg.norm_between_loops else h

    def logits(self, h_final: torch.Tensor) -> torch.Tensor:
        if not self.cfg.norm_between_loops:
            h_final = self.model.norm(h_final)
        return self.lm_head(h_final).float()

    def rope(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions: [T] long -> cos, sin [T, d]."""
        f = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


def load_ouro(name_or_path: str = "ByteDance/Ouro-1.4B", device="cuda", dtype=torch.bfloat16):
    """Load an Ouro checkpoint into ``LoopedLM`` without HF remote code. Returns (model, tokenizer)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    path = Path(name_or_path)
    if not path.exists():
        path = Path(snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    cfg = LoopedConfig.from_hf(path)
    with torch.device("meta"):
        model = LoopedLM(cfg)
    state = {}
    for f in sorted(path.glob("*.safetensors")):
        state.update(load_file(str(f), device="cpu"))
    model.load_state_dict(state, strict=True, assign=True)
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    model.inv_freq = inv
    model = model.to(device=device, dtype=dtype).eval()
    model.inv_freq = inv.to(device)  # keep RoPE frequencies in fp32
    tok = AutoTokenizer.from_pretrained(str(path))
    return model, tok


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def init_tiny(cfg: LoopedConfig, seed: int = 0) -> LoopedLM:
    torch.manual_seed(seed)
    m = LoopedLM(cfg)
    for p in m.parameters():
        if p.dim() == 2:
            nn.init.normal_(p, std=0.02 / math.sqrt(1 + cfg.num_hidden_layers / 4))
    return m
