"""WikiText-2 continuation perplexity: dense vs FlashLoop ablations and variants.

Mirrors the paper's long-context protocol (App. F): deterministic prefixes,
NLL over the following continuation tokens, scored through the decode path so
sparse attention and KV quantization are exercised.

    python scripts/eval_ppl.py --prefix 2048 --cont 64 --n 8
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashloop import DenseEngine, FlashLoopConfig, FlashLoopEngine, QuantConfig, load_ouro  # noqa: E402
from flashloop.data import wikitext2_ids, windows  # noqa: E402


def methods(model_key: str):
    p = FlashLoopConfig.paper(model_key)
    q = QuantConfig()
    dense_cols = [1.0] * len(p.col_keep)
    return {
        "dense": None,
        "token-sparse": replace(p, col_keep=dense_cols, quant=None),
        "+sparse-attn": replace(p, quant=None),
        "FlashLoop (paper)": p,
        "per-loop KIVI4 (ablation)": replace(p, quant=replace(q, residual=False)),
        "FlashLoop INT2": replace(p, quant=replace(q, bits=2)),
        "V0 keep-last-token": replace(p, keep_last_token_active=True),
        "V2 exit-gate score": replace(p, token_select="gate"),
        "V1 energy budget": replace(p, token_select="energy", token_energy=0.25, token_keep=[1.0, 1.0, 0.5, 0.5]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
    ap.add_argument("--key", default="ouro-1.4b")
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results/ppl.json")
    a = ap.parse_args()

    model, tok = load_ouro(a.model)
    ids = wikitext2_ids(tok)
    wins = windows(ids, a.prefix, a.cont, a.n)
    res = {}
    for name, cfg in methods(a.key).items():
        if a.only and name not in a.only:
            continue
        eng = DenseEngine(model) if cfg is None else FlashLoopEngine(model, cfg)
        nll, kv, act, t0 = [], [], None, time.time()
        for pre, cont in wins:
            nll.append(eng.score_continuation(pre.cuda(), cont.cuda()).cpu())
            kv.append(eng.kv_bytes())
            act = getattr(eng.stats, "active_per_loop", None)
        nll = torch.cat(nll)
        res[name] = dict(ppl=nll.mean().exp().item(), nll=nll.mean().item(), kv_mib=sum(kv) / len(kv) / 2**20,
                         active=act, sec=time.time() - t0)
        print(f"{name:28s} ppl {res[name]['ppl']:7.3f}  KV {res[name]['kv_mib']:8.1f} MiB  "
              f"active {['%.2f' % x for x in act or []]}  {res[name]['sec']:.0f}s", flush=True)
        del eng
        torch.cuda.empty_cache()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(args=vars(a), results=res), indent=2))  # incremental
    base = res.get("dense")
    if base:
        for r in res.values():
            r["kv_reduction"] = base["kv_mib"] / r["kv_mib"]
            r["dppl"] = r["ppl"] - base["ppl"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(args=vars(a), results=res), indent=2))


if __name__ == "__main__":
    main()
