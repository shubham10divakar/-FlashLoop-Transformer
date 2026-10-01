"""Efficiency: KV-cache memory, prefill FLOPs and wall-clock vs context length (paper Fig. 6/7).

Latency numbers come from the pure-PyTorch reference engine (no fused kernels),
so they show algorithmic savings in prefill but understate decode speedups the
paper gets from custom CUDA kernels. Memory and FLOPs are exact accounting.

    python scripts/bench.py --ctx 1024 2048 4096 8192
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashloop import DenseEngine, FlashLoopConfig, FlashLoopEngine, load_ouro  # noqa: E402
from flashloop.data import wikitext2_ids  # noqa: E402


def timed(fn, reps=3):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B")
    ap.add_argument("--key", default="ouro-1.4b")
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--dec", type=int, default=16)
    ap.add_argument("--out", default="results/bench.json")
    a = ap.parse_args()
    model, tok = load_ouro(a.model)
    ids = wikitext2_ids(tok)
    rows = []
    for T in a.ctx:
        x = ids[:T].cuda()[None]
        for name in ("dense", "flashloop"):
            eng = DenseEngine(model) if name == "dense" else FlashLoopEngine(model, FlashLoopConfig.paper(a.key))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                t_pre, _ = timed(lambda: eng.prefill(x))
                flops = eng.stats.flops
                for i in range(a.dec):  # warm-up
                    eng.decode(int(ids[T + i]))
                torch.cuda.synchronize()
                t = time.perf_counter()
                for i in range(a.dec):
                    eng.decode(int(ids[T + a.dec + i]))
                torch.cuda.synchronize()
                t_dec = (time.perf_counter() - t) / a.dec
                row = dict(ctx=T, method=name, prefill_ms=t_pre * 1e3, decode_ms=t_dec * 1e3,
                           prefill_tflops=flops / 1e12, kv_mib=eng.kv_bytes() / 2**20,
                           peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            except torch.OutOfMemoryError:
                row = dict(ctx=T, method=name, oom=True)
            rows.append(row)
            print(row, flush=True)
            del eng
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
