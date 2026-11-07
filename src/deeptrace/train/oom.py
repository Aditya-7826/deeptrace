"""CUDA out-of-memory fallback: halve the batch size and retry once."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

import torch

LOG = logging.getLogger("deeptrace.train")
T = TypeVar("T")


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)


def run_with_oom_fallback(
    fn: Callable[[int], T],
    batch_size: int,
    on_oom: Callable[[int], None],
    allow_retry: bool = True,
) -> tuple[T, int, bool]:
    """Run fn(batch_size). On CUDA OOM, call on_oom(new_bs) to restore state, then
    retry once with half the batch size. Returns (result, batch_size_used, retried)."""
    try:
        return fn(batch_size), batch_size, False
    except BaseException as exc:
        if not (allow_retry and is_cuda_oom(exc)) or batch_size <= 1:
            raise
        new_bs = max(1, batch_size // 2)
        LOG.warning(
            "!!! CUDA out of memory at batch size %d. Halving to %d and retrying once. !!!",
            batch_size,
            new_bs,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        on_oom(new_bs)
        return fn(new_bs), new_bs, True
