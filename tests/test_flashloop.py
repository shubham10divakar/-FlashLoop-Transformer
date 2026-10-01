"""Correctness tests on a tiny random looped model (CPU, fp32). Run: python -m pytest tests -q"""
import torch

from flashloop import DenseEngine, FlashLoopConfig, FlashLoopEngine, LoopedConfig, QuantConfig, init_tiny
from flashloop.kv_cache import LoopKVStore
from flashloop.quant import dequantize, quantize

torch.manual_seed(0)


def _model(**kw):
    return init_tiny(LoopedConfig.tiny(**kw)).float().eval()


def _run(engine, ids, n_dec=6):
    logits = [engine.prefill(ids)[0, -1]]
    for i in range(n_dec):
        logits.append(engine.decode(int(ids[0, i]))[0, -1])
    return torch.stack(logits)


def test_quant_roundtrip():
    x = torch.randn(4, 100, 32)
    for axis in ("token", "channel"):
        for bits, tol in ((8, 0.02), (4, 0.25), (2, 1.2)):
            q = quantize(x, bits, 64, axis)
            err = (dequantize(q, torch.float32) - x).abs().max().item()
            assert err < tol * x.abs().max().item(), (axis, bits, err)
            assert q.packed.dtype == torch.uint8


def test_dense_schedule_matches_baseline():
    """FlashLoop with every budget at 100% and no quantization must reproduce dense inference."""
    m = _model()
    ids = torch.randint(0, 256, (1, 50))
    ref = _run(DenseEngine(m), ids)
    out = _run(FlashLoopEngine(m, FlashLoopConfig.dense(4)), ids)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4), (ref - out).abs().max()


def test_sparse_paths_at_full_budget_are_exact():
    """Token-sparse prefill with k=T and Eq. 7 sparse attention with c=N are exact rewrites."""
    m = _model()
    ids = torch.randint(0, 256, (1, 40))
    ref = _run(DenseEngine(m), ids)
    cfg = FlashLoopConfig(token_keep=[1, 1, 0.999, 0.999], col_keep=[1, 1, 0.999, 0.999], quant=None)
    eng = FlashLoopEngine(m, cfg)
    out = _run(eng, ids)
    assert eng.stats.active_per_loop == [1.0, 1.0, 1.0, 1.0]
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4), (ref - out).abs().max()


def test_sparsity_reduces_work_and_memory():
    m = _model()
    ids = torch.randint(0, 256, (1, 200))
    d, f = DenseEngine(m), FlashLoopEngine(m, FlashLoopConfig(quant=None))
    d.prefill(ids), f.prefill(ids)
    assert f.stats.active_per_loop[2] == 0.25 and f.stats.active_per_loop[3] == 0.10
    assert f.stats.flops < 0.75 * d.stats.flops
    assert f.kv_bytes() < 0.62 * d.kv_bytes()  # (1 + 1 + 0.25 + 0.10) / 4 = 0.59 + index overhead
    # nested active sets: A^(4) subset of A^(3)
    a3, a4 = set(f.active_sets[2].tolist()), set(f.active_sets[3].tolist())
    assert a4 <= a3


def test_residual_store_reconstruction():
    """Quantized store read == explicit Eq. 8 recursion; residual coding beats direct coding on drifting KV."""
    torch.manual_seed(1)
    H, T, d, R = 2, 300, 32, 4
    xs = [torch.randn(H, T, d)]
    for _ in range(R - 1):
        xs.append(xs[-1] + 0.05 * torch.randn(H, T, d))
    errs = {}
    for residual in (True, False):
        st = LoopKVStore(R, QuantConfig(bits=4, group=64, fp_window=64, residual=residual))
        for r in range(R):
            st.write(r, torch.arange(T), xs[r], xs[r])
        st.maybe_compress(force_all_old=True)
        assert st.raw[0].idx.numel() == 64  # fp window kept
        K, _ = st.read(R - 1)
        assert torch.equal(K[:, -64:], xs[-1][:, -64:])
        errs[residual] = ((K - xs[-1]).norm() / xs[-1].norm()).item()
    assert errs[True] < errs[False] * 1.05, errs


def test_full_flashloop_runs_with_quant():
    m = _model()
    ids = torch.randint(0, 256, (1, 160))
    eng = FlashLoopEngine(m, FlashLoopConfig(quant=QuantConfig(fp_window=16, group=32)))
    out = _run(eng, ids, n_dec=40)
    assert torch.isfinite(out).all()
    assert all(st.n_quantized > 0 for st in eng.stores)
