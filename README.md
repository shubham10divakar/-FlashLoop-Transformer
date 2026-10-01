# FlashLoop — reference implementation

Training-free fast, memory-efficient inference for looped Transformers, after
*FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates*
(Yang & Liu, arXiv:2609.29812). Pure PyTorch, batch size 1, runs real
`ByteDance/Ouro-1.4B` weights without HF remote code.

| Component | Where | Paper |
| --- | --- | --- |
| Cross-loop token-sparse updates (prefill) | `FlashLoopEngine.prefill`, `_select_tokens` | Sec 4.1, Eq. 5 |
| Loop-aware sparse attention w/ cached global mass (decode) | `FlashLoopEngine.decode`, `_pick_columns` | Sec 4.2, Eq. 6–7 |
| Cross-loop KV residual quantization (INT4, g=64, fp window 64) | `LoopKVStore`, `quant.py` | Sec 4.3, Eq. 4, 8 |

```bash
python -m pytest tests -q                          # exactness + quantizer tests (CPU, tiny model)
python scripts/analyze_redundancy.py --T 1024 --n 4  # Sec. 3 measurements on Ouro-1.4B
python scripts/eval_ppl.py --prefix 2048 --cont 64 --n 8
python scripts/bench.py --ctx 1024 2048 4096 8192
```

```python
from flashloop import load_ouro, FlashLoopEngine, FlashLoopConfig
model, tok = load_ouro("ByteDance/Ouro-1.4B")
eng = FlashLoopEngine(model, FlashLoopConfig.paper("ouro-1.4b"))
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
print(tok.decode(eng.generate(ids, 32)))
```

Research variants are config flags on `FlashLoopConfig` (`token_select="energy"`,
`keep_last_token_active`, `QuantConfig(residual=False, bits=2)`); see the design doc.

Notes
- `kv_bytes()` = packed logical footprint. The reference memoizes dequantized
  history for speed (`QuantConfig.cache_dequant`); disable it to measure true peak memory.
- Latency is reference PyTorch: prefill savings are real, decode lacks the paper's fused kernels.
