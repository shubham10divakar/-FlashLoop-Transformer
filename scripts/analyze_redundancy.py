"""Measure the three cross-loop redundancies (paper Sec. 3, Fig. 2/3) on a real looped model.

    python scripts/analyze_redundancy.py --T 1024 --n 4
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashloop import DenseEngine, load_ouro  # noqa: E402
from flashloop.data import wikitext2_ids, windows  # noqa: E402
from flashloop.quant import fake_quant  # noqa: E402


def rel(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dec", type=int, default=8)
    ap.add_argument("--out", default="results/redundancy.json")
    a = ap.parse_args()
    model, tok = load_ouro(a.model)
    R, L = model.cfg.total_ut_steps, model.cfg.num_hidden_layers
    wins = windows(wikitext2_ids(tok), a.T, a.dec, a.n, seed=1)

    out = {"token_delta_quantiles": {}, "frac_tokens_above": {}, "top_frac_change_share": {},
           "nested_overlap_top10": [], "col_top10_mass": [[] for _ in range(R)],
           "col_turnover_top10": [[] for _ in range(R - 1)], "attn_recon_err_vs_keep": {},
           "kv_residual_rms": {"K": [[] for _ in range(R - 1)], "V": [[] for _ in range(R - 1)]},
           "quant_err": {"direct": [[] for _ in range(R)], "residual": [[] for _ in range(R)]}}
    deltas = [[] for _ in range(R - 1)]
    keeps = [0.05, 0.10, 0.20, 0.40, 0.80]
    recon = {kp: [[] for _ in range(R - 1)] for kp in keeps}

    for pre, cont in wins:
        probes = {}
        eng = DenseEngine(model, record_hidden=True,
                          probe=lambda r, l, q, K, V: probes.__setitem__((r, l), (q[0, :, 0].float(), K[0].float(), V[0].float())))
        eng.prefill(pre.cuda())
        Hs = eng.loop_hidden  # R x [T, D]
        # ---- 3.1 token-update redundancy
        for r in range(1, R):
            s = (Hs[r] - Hs[r - 1]).norm(dim=-1) / (Hs[r].norm(dim=-1) + 1e-6)
            deltas[r - 1].append(s)
        # nested structure: overlap of top-10% changers in consecutive transitions
        for r in range(1, R - 1):
            k = int(0.1 * a.T)
            t1 = set(deltas[r - 1][-1].topk(k).indices.tolist())
            t2 = set(deltas[r][-1].topk(k).indices.tolist())
            out["nested_overlap_top10"].append(len(t1 & t2) / k)
        # ---- 3.3 KV residuals (prefill cache, every layer)
        for l in range(L):
            for name, store in (("K", eng.cache.k), ("V", eng.cache.v)):
                xs = [store[(r, l)][0].float() for r in range(R)]
                for r in range(1, R):
                    out["kv_residual_rms"][name][r - 1].append(rel(xs[r], xs[r - 1]))  # RMS(delta) / RMS(base)
                if l % 4 == 0 and name == "K":  # INT4 direct vs residual chain (Fig. 2c), subsample layers
                    prev = None
                    for r in range(R):
                        out["quant_err"]["direct"][r].append(rel(fake_quant(xs[r], 4, 64, "token"), xs[r]))
                        prev = fake_quant(xs[r], 4, 64, "token") if prev is None else prev + fake_quant(xs[r] - prev, 4, 64, "token")
                        out["quant_err"]["residual"][r].append(rel(prev, xs[r]))
        # ---- 3.2 attention-column redundancy (decode queries)
        for i in range(a.dec):
            probes.clear()
            eng.decode(int(cont[0, i]))
            for l in range(0, L, 2):
                ps, os_, vs = [], [], []
                for r in range(R):
                    q, K, V = probes[(r, l)]
                    g = q.shape[0] // K.shape[0]
                    K, V = K.repeat_interleave(g, 0), V.repeat_interleave(g, 0)
                    p = torch.softmax((K @ q[..., None]).squeeze(-1) / q.shape[-1] ** 0.5, -1)  # [H, N]
                    ps.append(p), vs.append(V), os_.append((p[..., None] * V).sum(1))
                    N = p.shape[-1]
                    k10 = max(1, int(0.1 * N))
                    out["col_top10_mass"][r].append(p.topk(k10, -1).values.sum(-1).mean().item())
                for r in range(1, R):
                    N = ps[r].shape[-1]
                    k10 = max(1, int(0.1 * N))
                    a_ = ps[r - 1].topk(k10, -1).indices
                    b_ = ps[r].topk(k10, -1).indices
                    inter = (a_[..., :, None] == b_[..., None, :]).any(-1).float().mean().item()
                    out["col_turnover_top10"][r - 1].append(1 - inter)
                    # Eq. 3 reconstruction with columns ranked in loop r-1
                    for kp in keeps:
                        kk = max(1, int(kp * N))
                        S = ps[r - 1].topk(kk, -1).indices
                        pprev = ps[r - 1].gather(-1, S)
                        m = pprev.sum(-1, keepdim=True)
                        gi = S[..., None].expand(-1, -1, vs[r].shape[-1])
                        q, K, _ = probes[(r, l)]
                        Kr = K.repeat_interleave(q.shape[0] // K.shape[0], 0).gather(1, gi)
                        pnew = m * torch.softmax((Kr @ q[..., None]).squeeze(-1) / q.shape[-1] ** 0.5, -1)
                        o_hat = os_[r - 1] - (pprev[..., None] * vs[r - 1].gather(1, gi)).sum(1) + (pnew[..., None] * vs[r].gather(1, gi)).sum(1)
                        recon[kp][r - 1].append(rel(o_hat, os_[r]))
        del eng
        torch.cuda.empty_cache()

    mean = lambda xs: sum(xs) / len(xs)
    for r in range(1, R):
        s = torch.cat(deltas[r - 1])
        tr = f"{r}->{r + 1}"
        out["token_delta_quantiles"][tr] = {q: s.quantile(q).item() for q in (0.1, 0.5, 0.75, 0.9, 0.99)}
        out["frac_tokens_above"][tr] = {t: (s > t).float().mean().item() for t in (0.01, 0.05, 0.1, 0.2)}
        e = s.pow(2).sort(descending=True).values
        out["top_frac_change_share"][tr] = {f: (e[: int(f * e.numel())].sum() / e.sum()).item() for f in (0.1, 0.25)}
    out["nested_overlap_top10"] = mean(out["nested_overlap_top10"])
    out["col_top10_mass"] = [mean(x) for x in out["col_top10_mass"]]
    out["col_turnover_top10"] = [mean(x) for x in out["col_turnover_top10"]]
    out["attn_recon_err_vs_keep"] = {kp: [mean(x) for x in recon[kp]] for kp in keeps}
    for k in ("K", "V"):
        out["kv_residual_rms"][k] = [mean(x) for x in out["kv_residual_rms"][k]]
    for k in ("direct", "residual"):
        out["quant_err"][k] = [mean(x) for x in out["quant_err"][k]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
