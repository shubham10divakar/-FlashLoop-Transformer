"""Asymmetric group-wise integer quantization with real bit-packing.

Paper setting (App. E): INT4, asymmetric, group size 64; keys per-channel
(groups run along tokens, KIVI-style, post-RoPE), values per-token (groups run
along channels).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QTensor:
    packed: torch.Tensor  # uint8, bit-packed codes
    scale: torch.Tensor  # fp16, one per group
    zero: torch.Tensor  # fp16 (group min), one per group
    shape: tuple  # original shape [Hkv, n, d]
    axis: str  # "token" (per-channel keys) or "channel" (per-token values)
    bits: int
    group: int

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scale, self.zero))


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    codes = codes.to(torch.uint8).flatten()
    per = 8 // bits
    pad = (-codes.numel()) % per
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    codes = codes.view(-1, per)
    out = torch.zeros(codes.shape[0], dtype=torch.uint8, device=codes.device)
    for i in range(per):
        out |= codes[:, i] << (bits * i)
    return out


def _unpack(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    per = 8 // bits
    mask = (1 << bits) - 1
    cols = [(packed >> (bits * i)) & mask for i in range(per)]
    return torch.stack(cols, dim=1).flatten()[:numel]


def _to_groups(x: torch.Tensor, axis: str, group: int) -> tuple[torch.Tensor, int]:
    """x: [H, n, d] -> [num_groups, group] plus the padded length along the grouped axis."""
    if axis == "token":  # per-channel: group consecutive tokens of one channel
        x = x.transpose(1, 2)  # [H, d, n]
    H, a, n = x.shape
    pad = (-n) % group
    if pad:  # replicate last element so padding does not widen the [min,max] range
        x = torch.cat([x, x[..., -1:].expand(H, a, pad)], dim=-1)
    return x.reshape(-1, group), n + pad


def quantize(x: torch.Tensor, bits: int = 4, group: int = 64, axis: str = "token") -> QTensor:
    assert bits in (2, 4, 8)
    shape = tuple(x.shape)
    g, _ = _to_groups(x.float(), axis, group)
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((mx - mn) / (2**bits - 1)).clamp_min(1e-8)
    codes = torch.round((g - mn) / scale).clamp_(0, 2**bits - 1)
    return QTensor(_pack(codes, bits), scale.half(), mn.half(), shape, axis, bits, group)


def dequantize(q: QTensor, dtype=torch.bfloat16) -> torch.Tensor:
    H, n, d = q.shape
    a, L = (d, n) if q.axis == "token" else (n, d)
    Lp = L + (-L) % q.group
    codes = _unpack(q.packed, q.bits, H * a * Lp).view(-1, q.group).float()
    x = (codes * q.scale.float() + q.zero.float()).view(H, a, Lp)[..., :L]
    if q.axis == "token":
        x = x.transpose(1, 2)
    return x.to(dtype).contiguous()


def fake_quant(x: torch.Tensor, bits=4, group=64, axis="token") -> torch.Tensor:
    return dequantize(quantize(x, bits, group, axis), x.dtype)
