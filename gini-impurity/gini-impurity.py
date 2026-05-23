import numpy as np
import torch

def gini_impurity(y_left, y_right):
    def gini(y):
        if len(y) == 0:
            return 0.0
        t = torch.tensor(y, dtype=torch.long)
        t = t - t.min()
        counts = torch.bincount(t).double()
        p = counts / counts.sum()
        return (1 - torch.sum(p ** 2)).item()

    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    return (n_left / n_total) * gini(y_left) + (n_right / n_total) * gini(y_right)