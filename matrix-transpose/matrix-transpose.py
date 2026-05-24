
import numpy as np
import torch

def matrix_transpose(A):
    return torch.tensor(A).T.numpy()
