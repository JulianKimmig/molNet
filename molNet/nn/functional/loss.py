import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.nn import _reduction as _Reduction


def rell1_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()
            ),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    target[target == 0] = 10 ** -32
    ret = torch.abs((input - target) / target)
    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    return ret


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class CrossEntropyLossFromOneHot(torch.nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super(CrossEntropyLossFromOneHot, self).forward(input,torch.argmax(target, dim=1))
