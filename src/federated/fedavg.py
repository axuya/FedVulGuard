from collections import OrderedDict
import torch


@torch.no_grad()
def fedavg(state_dicts, weights):
    """
    Standard FedAvg aggregation.

    Args:
        state_dicts: List[OrderedDict[str, Tensor]]
        weights:     List[float]  (e.g., client sample counts)

    Returns:
        OrderedDict[str, Tensor]: aggregated state dict
    """
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total = float(sum(weights))
    w = [float(x) / total for x in weights]

    agg = OrderedDict()
    keys = list(state_dicts[0].keys())

    for k in keys:
        # Non-float buffers (e.g., BN num_batches_tracked) -> take from first client
        if not torch.is_floating_point(state_dicts[0][k]):
            agg[k] = state_dicts[0][k]
            continue

        acc = None
        for sd, wi in zip(state_dicts, w):
            t = sd[k].detach()
            acc = t * wi if acc is None else acc + t * wi
        agg[k] = acc

    return agg
