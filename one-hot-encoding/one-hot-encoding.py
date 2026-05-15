import numpy as np

import torch
import torch.nn.functional as F

def one_hot(y, num_classes=None):
    y = torch.as_tensor(y, dtype=torch.long)
    if num_classes is None:
        num_classes = y.max().item() + 1
    return F.one_hot(y, num_classes=num_classes)