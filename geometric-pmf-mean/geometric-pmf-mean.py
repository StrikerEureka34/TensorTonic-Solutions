import numpy as np
import torch

def geometric_pmf_mean(k, p):
    k_tensor = torch.tensor(k, dtype=torch.float64)
    pmf = (1 - p) ** (k_tensor - 1) * p
    mean = 1 / p
    return pmf.numpy(), mean