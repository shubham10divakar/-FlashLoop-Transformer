"""Evaluation data helpers (WikiText-2, as used for the paper's calibration and long-context PPL)."""
from __future__ import annotations

import torch


def wikitext2_ids(tok, split: str = "test") -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]


def windows(ids: torch.Tensor, prefix_len: int, cont_len: int, n: int, seed: int = 0):
    """``n`` deterministic (prefix, continuation) windows from a token stream."""
    g = torch.Generator().manual_seed(seed)
    span = prefix_len + cont_len
    starts = torch.randint(0, ids.numel() - span, (n,), generator=g)
    return [(ids[s:s + prefix_len][None], ids[s + prefix_len:s + span][None]) for s in starts.tolist()]
