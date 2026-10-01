"""Dense looped inference (baseline) and FlashLoop inference (paper Sec. 4 / App. D).

Both engines are batch-size-1 reference implementations in plain PyTorch. They
share the model primitives in ``model.py`` so any difference in outputs comes
from FlashLoop's approximations alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .kv_cache import DenseLoopCache, LoopKVStore, QuantConfig
from .model import LoopedLM


# ============================================================================ configs
@dataclass
class FlashLoopConfig:
    """Loop-wise schedule. Lists are indexed by loop (0-based); ``None``/1.0 = dense."""

    token_keep: list = field(default_factory=lambda: [1.0, 1.0, 0.25, 0.10])  # prefill token retention
    col_keep: list = field(default_factory=lambda: [1.0, 1.0, 0.10, 0.10])  # decode column retention
    quant: QuantConfig | None = field(default_factory=QuantConfig)
    eps: float = 1e-6
    # --- implementation choices the paper leaves open (defaults = closest to the paper) ---
    keep_last_token_active: bool = False  # force the final prompt token into every active set
    keep_self_column: bool = True  # always recompute the decode token's own key column
    # --- research variants (see docs) ---
    token_select: str = "topk"  # "topk" (paper) | "energy" (V1: prompt-adaptive budget) | "gate" (V2)
    token_energy: float = 0.5  # V1: keep the fewest tokens covering this share of sum(s_i^2)
    token_min_keep: float = 0.02  # V1: floor on the active fraction (of T)

    @classmethod
    def dense(cls, n_loops=4):
        return cls(token_keep=[1.0] * n_loops, col_keep=[1.0] * n_loops, quant=None)

    @classmethod
    def paper(cls, model: str = "ouro-1.4b", quant: bool = True):
        """Table 5 schedules."""
        sched = {
            "ouro-1.4b": ([1, 1, 0.25, 0.10], [1, 1, 0.10, 0.10]),
            "ouro-1.4b-thinking": ([1, 1, 0.25, 0.10], [1, 1, 0.12, 0.12]),
            "ouro-2.6b": ([1, 1, 0.20, 0.08], [1, 1, 0.08, 0.08]),
            "ouro-2.6b-thinking": ([1, 1, 0.20, 0.08], [1, 1, 0.08, 0.08]),
        }
        if model.startswith("huginn"):
            tk = [1.0] * 4 + [0.9 + (0.2 - 0.9) * i / 27 for i in range(28)]
            ck = [1.0] * 4 + [0.76 + (0.10 - 0.76) * i / 27 for i in range(28)]
        else:
            tk, ck = sched[model]
        return cls(token_keep=list(tk), col_keep=list(ck), quant=QuantConfig() if quant else None)


def _is_dense(x) -> bool:
    return x is None or x >= 1.0


@dataclass
class Stats:
    flops: float = 0.0  # multiply-adds x2, transformer blocks only (lm_head excluded)
    rows: int = 0  # token-rows pushed through a layer
    active_per_loop: list = field(default_factory=list)

    def reset(self):
        self.flops, self.rows, self.active_per_loop = 0.0, 0, []


class _Base:
    def __init__(self, model: LoopedLM):
        self.m = model
        c = model.cfg
        self.R, self.L = c.total_ut_steps, c.num_hidden_layers
        self.H, self.Hkv, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.scale = 1.0 / math.sqrt(self.d)
        D, I = c.hidden_size, c.intermediate_size
        self._lin_flops = 2 * (D * (self.H + 2 * self.Hkv) * self.d + self.H * self.d * D + 3 * D * I)
        self.stats = Stats()
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device

    def _count(self, n_rows: int, n_key_pairs: float):
        self.stats.flops += n_rows * self._lin_flops + 4 * self.H * self.d * n_key_pairs
        self.stats.rows += n_rows

    def _rep(self, x):  # [Hkv, N, d] -> [H, N, d]
        g = self.H // self.Hkv
        return x if g == 1 else x.repeat_interleave(g, dim=0)

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 64, eos: int | None = None):
        logits = self.prefill(ids)
        out = []
        for _ in range(max_new_tokens):
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            if eos is not None and nxt == eos:
                break
            logits = self.decode(nxt)
        return out

    @torch.no_grad()
    def score_continuation(self, prefix: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        """Per-token NLL of ``cont`` given ``prefix`` (teacher forcing through the decode path)."""
        logits = self.prefill(prefix)
        nll = []
        for i in range(cont.shape[-1]):
            nll.append(F.cross_entropy(logits[0, -1:], cont[..., i].view(1)))
            if i + 1 < cont.shape[-1]:
                logits = self.decode(int(cont[..., i]))
        return torch.stack(nll)


# ============================================================================ dense
class DenseEngine(_Base):
    """Original looped inference: every loop updates every token and caches its own K/V."""

    def __init__(self, model: LoopedLM, record_hidden: bool = False, probe=None):
        super().__init__(model)
        self.record_hidden = record_hidden
        self.probe = probe  # optional callback(r, l, q, K, V) during decode, for analysis

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        self.stats.reset()
        self.cache = DenseLoopCache()
        T = ids.shape[-1]
        pos = torch.arange(T, device=self.device)
        cos, sin = self.m.rope(pos, self.dtype)
        h = self.m.embed(ids.view(1, T))
        self.loop_hidden = []
        for r in range(self.R):
            for l, layer in enumerate(self.m.layers):
                q, k, v = layer.qkv(h, cos, sin)
                K, V = self.cache.append((r, l), k, v)
                o = F.scaled_dot_product_attention(q, K, V, is_causal=True, enable_gqa=self.H != self.Hkv)
                h = layer.finish(h, o)
                self._count(T, T * (T + 1) / 2)
            h = self.m.end_of_loop(h)
            if self.record_hidden:
                self.loop_hidden.append(h[0].float().cpu())
        self.T = T
        return self.m.logits(h[:, -1:])

    @torch.no_grad()
    def decode(self, token: int) -> torch.Tensor:
        t = self.T
        cos, sin = self.m.rope(torch.tensor([t], device=self.device), self.dtype)
        h = self.m.embed(torch.tensor([[token]], device=self.device))
        for r in range(self.R):
            for l, layer in enumerate(self.m.layers):
                q, k, v = layer.qkv(h, cos, sin)
                K, V = self.cache.append((r, l), k, v)
                if self.probe is not None:
                    self.probe(r, l, q, K, V)
                o = F.scaled_dot_product_attention(q, K, V, enable_gqa=self.H != self.Hkv)
                h = layer.finish(h, o)
                self._count(1, t + 1)
            h = self.m.end_of_loop(h)
        self.T = t + 1
        return self.m.logits(h)

    def kv_bytes(self) -> int:
        return self.cache.nbytes()


# ============================================================================ FlashLoop
class FlashLoopEngine(_Base):
    def __init__(self, model: LoopedLM, cfg: FlashLoopConfig | None = None):
        super().__init__(model)
        self.cfg = cfg or FlashLoopConfig.paper()
        assert len(self.cfg.token_keep) == self.R and len(self.cfg.col_keep) == self.R, "schedule length != loops"

    # --------------------------------------------------------- token selection (D.1)
    def _select_tokens(self, H1, H2, active, r, T):
        """H1 = H^(r-1), H2 = H^(r-2) ([T, D]); returns the nested active set A^(r) (sorted)."""
        c = self.cfg
        a1, a2 = H1[active].float(), H2[active].float()
        s = (a1 - a2).norm(dim=-1) / (a1.norm(dim=-1) + c.eps)
        if c.token_select == "gate":  # V2: Ouro's learned exit gate; keep tokens least ready to exit
            gate = self.m.model.early_exit_gate
            s = 1 - torch.sigmoid(gate(H1[active].to(gate.weight.dtype)).float().squeeze(-1))
        if c.token_select == "energy":
            e = s.pow(2)
            order = e.argsort(descending=True)
            k = int((e[order].cumsum(0) < c.token_energy * e.sum()).sum()) + 1
            k = max(k, math.ceil(c.token_min_keep * T))
            if not _is_dense(c.token_keep[r]):
                k = min(k, math.ceil(c.token_keep[r] * T))  # budget acts as a cap
            sel = order[: min(k, active.numel())]
        else:
            k = min(active.numel(), max(1, math.ceil(c.token_keep[r] * T)))
            if c.keep_last_token_active and bool((active == T - 1).any()):
                s = s.clone()
                s[active == T - 1] = float("inf")
            sel = s.topk(k).indices
        return active[sel].sort().values

    # ------------------------------------------------------------------- prefill (4.1)
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        self.stats.reset()
        T = ids.shape[-1]
        self.stores = [LoopKVStore(self.R, self.cfg.quant) for _ in range(self.L)]
        pos = torch.arange(T, device=self.device)
        cos, sin = self.m.rope(pos, self.dtype)
        h = self.m.embed(ids.view(1, T))
        hist = []  # loop outputs H^(r), [T, D]
        active = pos
        self.active_sets = []
        for r in range(self.R):
            sparse = len(hist) >= 2 and not _is_dense(self.cfg.token_keep[r])
            if sparse:
                active = self._select_tokens(hist[-1], hist[-2], active, r, T)
            else:
                active = pos
            self.active_sets.append(active)
            self.stats.active_per_loop.append(active.numel() / T)
            if not sparse:
                for l, layer in enumerate(self.m.layers):
                    q, k, v = layer.qkv(h, cos, sin)
                    self.stores[l].write(r, pos, k[0], v[0])  # dense loop overwrites every row
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.H != self.Hkv)
                    h = layer.finish(h, o)
                    self._count(T, T * (T + 1) / 2)
                h = self.m.end_of_loop(h)
            else:
                hA = h[:, active]
                cA, sA = cos[active], sin[active]
                mask = (pos[None, :] <= active[:, None])[None, None]  # [1,1,n,T]
                pairs = float((active + 1).sum())
                for l, layer in enumerate(self.m.layers):
                    q, k, v = layer.qkv(hA, cA, sA)
                    st = self.stores[l]
                    st.write(r, active, k[0], v[0])  # inactive rows reuse loop r-1 K/V
                    K, V = st.read(r)
                    o = F.scaled_dot_product_attention(q, K[None], V[None], attn_mask=mask,
                                                       enable_gqa=self.H != self.Hkv)
                    hA = layer.finish(hA, o)
                    self._count(active.numel(), pairs)
                h = h.clone()
                h[:, active] = self.m.end_of_loop(hA)  # inactive tokens keep H^(r-1)
            hist.append(h[0])
            if len(hist) > 2:
                hist.pop(0)
        for st in self.stores:
            st.maybe_compress(force_all_old=True)
        self.T = T
        return self.m.logits(h[:, -1:])

    # ------------------------------------------------------------- decode (4.2, D.2)
    def _pick_columns(self, p, c_next, n_keys, cand=None):
        """p: [H, n] probs over candidate columns ``cand`` ([H, n] key indices, None = all)."""
        k = min(p.shape[-1], max(1, math.ceil(c_next * n_keys)))
        score = p
        if self.cfg.keep_self_column:
            is_self = (cand == n_keys - 1) if cand is not None else None
            score = p.clone()
            if is_self is None:
                score[:, -1] = float("inf")
            else:
                score[is_self] = float("inf")
        top = score.topk(k, dim=-1).indices  # [H, k] positions within candidates
        p_sel = p.gather(-1, top)
        S = top if cand is None else cand.gather(-1, top)
        return S, p_sel

    @torch.no_grad()
    def decode(self, token: int) -> torch.Tensor:
        t = self.T
        N = t + 1
        cos, sin = self.m.rope(torch.tensor([t], device=self.device), self.dtype)
        h = self.m.embed(torch.tensor([[token]], device=self.device))
        idx = torch.tensor([t], device=self.device)
        ctx = [None] * self.L  # per layer: (S [H,c], m [H,1], contrib [H,d], o [H,d])
        ck = self.cfg.col_keep
        for r in range(self.R):
            nxt_sparse = r + 1 < self.R and not _is_dense(ck[r + 1])
            for l, layer in enumerate(self.m.layers):
                q, k, v = layer.qkv(h, cos, sin)
                st = self.stores[l]
                st.write(r, idx, k[0], v[0])
                K, V = st.read(r)  # reference: full reconstruction, then gather
                K, V = self._rep(K).float(), self._rep(V).float()
                qh = q[0, :, 0].float()  # [H, d]
                if ctx[l] is None or _is_dense(ck[r]):
                    p = torch.softmax((K @ qh[..., None]).squeeze(-1) * self.scale, dim=-1)  # [H, N]
                    o = (p[..., None] * V).sum(1)
                    self._count(1, N)
                    if nxt_sparse:
                        S, pS = self._pick_columns(p, ck[r + 1], N)
                        m = pS.sum(-1, keepdim=True)
                        VS = V.gather(1, S[..., None].expand(-1, -1, self.d))
                        ctx[l] = (S, m, (pS[..., None] * VS).sum(1), o)
                else:
                    S, m, contrib_prev, o_prev = ctx[l]
                    gi = S[..., None].expand(-1, -1, self.d)
                    KS, VS = K.gather(1, gi), V.gather(1, gi)
                    pS = m * torch.softmax((KS @ qh[..., None]).squeeze(-1) * self.scale, dim=-1)
                    contrib = (pS[..., None] * VS).sum(1)
                    o = o_prev - contrib_prev + contrib  # Eq. 7
                    self._count(1, S.shape[-1])
                    if nxt_sparse:
                        S2, pS2 = self._pick_columns(pS, ck[r + 1], N, cand=S)
                        VS2 = V.gather(1, S2[..., None].expand(-1, -1, self.d))
                        ctx[l] = (S2, pS2.sum(-1, keepdim=True), (pS2[..., None] * VS2).sum(1), o)
                    else:
                        ctx[l] = (S, m, contrib, o)
                h = layer.finish(h, o.to(self.dtype)[None, :, None])
            h = self.m.end_of_loop(h)
        for st in self.stores:
            st.maybe_compress()
        self.T = N
        return self.m.logits(h)

    def kv_bytes(self) -> int:
        return sum(st.nbytes() for st in self.stores)
