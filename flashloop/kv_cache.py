"""Cross-loop KV storage for one decoder layer across all R loops (batch size 1).

Every write is ``(loop r, positions idx, K, V)`` for the tokens that were active
in loop r. Reading loop r reconstructs the full ``[Hkv, T, d]`` state:

    X^(1)        = written values (all tokens)
    X^(r)[i]     = written value          if i in A^(r)
                 = X^(r-1)[i]             otherwise          (token-sparse reuse, Eq. 8)

With quantization enabled, positions older than ``fp_window`` are re-encoded as
a quantized loop-1 base plus quantized residuals ``X^(r) - X_hat^(r-1)`` of the
active tokens of each later loop (Eq. 4 / Eq. 8). The newest ``fp_window``
tokens stay in full precision, as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .quant import QTensor, dequantize, quantize


@dataclass
class QuantConfig:
    bits: int = 4
    group: int = 64
    fp_window: int = 64  # most recent tokens kept in full precision
    residual: bool = True  # False = quantize each loop's full K/V independently (paper ablation)
    cache_dequant: bool = True  # reference-impl speedup: memoize dequantized history (costs real memory;
    #                             set False to measure true peak memory; nbytes() reports the packed size)


@dataclass
class _Raw:  # full-precision rows of one loop, positions sorted ascending
    idx: torch.Tensor  # [n] long
    k: torch.Tensor  # [Hkv, n, d]
    v: torch.Tensor


@dataclass
class _QChunk:  # quantized rows of one loop
    idx: torch.Tensor
    k: QTensor
    v: QTensor
    residual: bool


@dataclass
class LoopKVStore:
    n_loops: int
    quant: QuantConfig | None = None
    T: int = 0
    raw: list = field(default_factory=list)  # per loop: _Raw | None
    qchunks: list = field(default_factory=list)  # per loop: list[_QChunk]
    n_quantized: int = 0  # positions [0, n_quantized) live in qchunks
    dtype: torch.dtype = torch.bfloat16
    _qrec: list | None = None  # memoized dequantized [0, n_quantized) per loop

    def __post_init__(self):
        self.raw = [None] * self.n_loops
        self.qchunks = [[] for _ in range(self.n_loops)]

    # ------------------------------------------------------------------ write
    def write(self, r: int, idx: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Store loop-r K/V ([Hkv,n,d]) of positions ``idx`` (all >= n_quantized)."""
        self.T = max(self.T, int(idx.max()) + 1)
        self.dtype = k.dtype
        cur = self.raw[r]
        if cur is None:
            self.raw[r] = _Raw(idx, k, v)
            return
        if int(idx.min()) > int(cur.idx.max()):  # fast path: append (decode)
            self.raw[r] = _Raw(torch.cat([cur.idx, idx]), torch.cat([cur.k, k], 1), torch.cat([cur.v, v], 1))
            return
        allidx = torch.cat([cur.idx, idx])
        order = torch.argsort(allidx, stable=True)
        self.raw[r] = _Raw(allidx[order], torch.cat([cur.k, k], 1)[:, order], torch.cat([cur.v, v], 1)[:, order])

    # ------------------------------------------------------------------- read
    def read(self, r: int, upto_T: int | None = None):
        """Reconstructed (K_hat^(r), V_hat^(r)) of shape [Hkv, T, d]."""
        T = upto_T or self.T
        K = V = None
        if self.n_quantized:
            if self.quant.cache_dequant:
                if self._qrec is None:
                    self._qrec = self._dequant_history(self.n_loops - 1)
                qk, qv = self._qrec[r]
            else:
                qk, qv = self._dequant_history(r)[r]
            K, V = self._empty(qk, T), self._empty(qv, T)
            K[:, : self.n_quantized] = qk
            V[:, : self.n_quantized] = qv
        for s in range(r + 1):
            raw = self.raw[s]
            if raw is not None:
                if K is None:
                    K, V = self._empty(raw.k, T), self._empty(raw.v, T)
                K[:, raw.idx] = raw.k
                V[:, raw.idx] = raw.v
        return K, V

    def _dequant_history(self, r_max: int):
        """Eq. 8 over the quantized prefix: list of (K_hat^(s), V_hat^(s)) for s <= r_max."""
        out, K, V = [], None, None
        for s in range(r_max + 1):
            K = None if K is None else K.clone()
            V = None if V is None else V.clone()
            for c in self.qchunks[s]:
                dk, dv = dequantize(c.k, self.dtype), dequantize(c.v, self.dtype)
                if K is None:
                    K, V = self._empty(dk, self.n_quantized), self._empty(dv, self.n_quantized)
                if c.residual:
                    K[:, c.idx] += dk
                    V[:, c.idx] += dv
                else:
                    K[:, c.idx] = dk
                    V[:, c.idx] = dv
            out.append((K, V))
        return out

    @staticmethod
    def _empty(like, T):
        return torch.zeros(like.shape[0], T, like.shape[2], dtype=like.dtype, device=like.device)

    # --------------------------------------------------------------- compress
    def maybe_compress(self, force_all_old: bool = False):
        """Quantize positions that left the full-precision window (in groups)."""
        q = self.quant
        if q is None:
            return
        limit = self.T - q.fp_window
        if limit <= self.n_quantized:
            return
        if not force_all_old:  # during decode compress whole groups only
            n = (limit - self.n_quantized) // q.group * q.group
            if n == 0:
                return
            limit = self.n_quantized + n
        self._compress_range(self.n_quantized, limit)
        self.n_quantized = limit
        self._qrec = None

    def _compress_range(self, lo: int, hi: int):
        q = self.quant
        prev_k = prev_v = None  # X_hat^(r-1) at positions [lo, hi)  ([Hkv, hi-lo, d])
        for r in range(self.n_loops):
            raw = self.raw[r]
            if raw is None:
                continue
            sel = (raw.idx >= lo) & (raw.idx < hi)
            keep = ~sel
            idx, k, v = raw.idx[sel], raw.k[:, sel], raw.v[:, sel]
            self.raw[r] = _Raw(raw.idx[keep], raw.k[:, keep], raw.v[:, keep]) if bool(keep.any()) else None
            if idx.numel() == 0:
                continue
            loc = idx - lo
            use_res = q.residual and prev_k is not None
            tk = k - prev_k[:, loc] if use_res else k
            tv = v - prev_v[:, loc] if use_res else v
            qk = quantize(tk, q.bits, q.group, axis="token")
            qv = quantize(tv, q.bits, q.group, axis="channel")
            self.qchunks[r].append(_QChunk(idx, qk, qv, residual=use_res))
            dk, dv = dequantize(qk, k.dtype), dequantize(qv, v.dtype)
            if prev_k is None:
                prev_k = torch.zeros(k.shape[0], hi - lo, k.shape[2], dtype=k.dtype, device=k.device)
                prev_v = torch.zeros_like(prev_k)
            prev_k, prev_v = prev_k.clone(), prev_v.clone()
            if use_res:
                prev_k[:, loc] += dk
                prev_v[:, loc] += dv
            else:
                prev_k[:, loc] = dk
                prev_v[:, loc] = dv

    # ----------------------------------------------------------------- memory
    def nbytes(self) -> int:
        b = 0
        for raw in self.raw:
            if raw is not None:
                b += raw.k.numel() * raw.k.element_size() * 2 + raw.idx.numel() * 4  # int32 indices
        for chunks in self.qchunks:
            for c in chunks:
                b += c.k.nbytes() + c.v.nbytes() + c.idx.numel() * 4
        return b


class DenseLoopCache:
    """Baseline: a full K/V copy for every (loop, layer), as in the original looped model."""

    def __init__(self):
        self.k, self.v = {}, {}

    def append(self, key, k, v):
        if key in self.k:
            self.k[key] = torch.cat([self.k[key], k], 2)
            self.v[key] = torch.cat([self.v[key], v], 2)
        else:
            self.k[key], self.v[key] = k, v
        return self.k[key], self.v[key]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in (self.k, self.v) for t in d.values())
