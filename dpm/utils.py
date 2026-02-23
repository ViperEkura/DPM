import math
import torch
from torch import Tensor
from typing import List, Optional


def taylor_extrapolate(
    features_cache: List[Tensor], 
    order: int = 2,
    device: Optional[str] = None
) -> Tensor:
    """
    Predict next feature using Taylor expansion with finite difference derivatives.
    
    Equivalent to:
        P^k(x_t) = sum_{n=0}^{k} (1/n!)  sum_{m=0}^{n} (-1)^m  C(n, m)  f(x_{t - (1 + m)})
    
    Args:
        features_cache: [f(t-Δt), f(t-2Δt), ..., f(t-LΔt)] (L >= order + 1)
        order: Taylor expansion order (k)
        device: target device
    Returns:
        predicted_feature: f(t) approximation
    """
    if len(features_cache) < order + 1:
        raise ValueError(
            f"Cache size ({len(features_cache)}) insufficient for order {order}. "
            f"Need at least {order + 1} features."
        )
    
    # Determine device
    if device is None:
        device = features_cache[0].device
    
    # Move all needed features to device (only first order+1 are used)
    # features[m] = f(t - (m+1)Δt) for m = 0, 1, ..., order
    feats = [features_cache[-(m + 1)].to(device) for m in range(order + 1)]
    
    # Initialize prediction as zero tensor of same shape
    pred = torch.zeros_like(feats[0])

    # Outer sum over n = 0 to order
    for n in range(order + 1):
        inner_sum = torch.zeros_like(feats[0])
        # Inner sum over m = 0 to n
        for m in range(n + 1):
            coeff = ((-1) ** m) * math.comb(n, m)
            inner_sum += coeff * feats[m]  # feats[m] = f(t - (m+1)Δt)
        pred += inner_sum / math.factorial(n)
    
    return pred