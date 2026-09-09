"""Pure FedAvg aggregation logic - no sockets, no threads. Kept separate so
the numerics can be tested and trusted independently of the networking code."""

from typing import List, Optional

import numpy as np

from protocol import NUM_WEIGHTS


def fedavg(payloads: List[bytes], weights: Optional[List[float]] = None) -> bytes:
    """Element-wise average of a list of raw 224-byte wire payloads.

    weights: optional per-client weighting (e.g. proportional to each
    client's local sample count, for sample-weighted FedAvg). Defaults to a
    uniform average - i.e. classic FedAvg with equal client contribution,
    matching W_global = (W_A + W_B) / 2 for K=2.
    """
    if not payloads:
        raise ValueError("fedavg() requires at least one payload")

    arrays = [np.frombuffer(p, dtype='<f4') for p in payloads]
    for a in arrays:
        if a.shape[0] != NUM_WEIGHTS:
            raise ValueError(f"every payload must contain exactly {NUM_WEIGHTS} float32 values")

    stacked = np.stack(arrays, axis=0).astype(np.float64)  # (K, NUM_WEIGHTS), fp64 accumulation

    if weights is None:
        averaged = stacked.mean(axis=0)
    else:
        if len(weights) != len(payloads):
            raise ValueError("weights must have the same length as payloads")
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        averaged = (stacked * w[:, None]).sum(axis=0)

    return averaged.astype('<f4').tobytes()


def payload_norm(payload: bytes) -> float:
    """L2 norm of a wire payload - used for lightweight per-round logging."""
    return float(np.linalg.norm(np.frombuffer(payload, dtype='<f4')))


def payload_delta_norm(a: bytes, b: bytes) -> float:
    """L2 norm of the difference between two payloads - a simple convergence proxy."""
    va = np.frombuffer(a, dtype='<f4')
    vb = np.frombuffer(b, dtype='<f4')
    return float(np.linalg.norm(va - vb))
