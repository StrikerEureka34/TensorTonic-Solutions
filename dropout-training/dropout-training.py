import numpy as np
import torch

def dropout(x, p=0.5, rng=None):
    t = torch.tensor(x, dtype=torch.float64)
    
    if rng is not None:
        mask = torch.tensor(rng.random(t.shape) >= p, dtype=torch.float64)
    else:
        mask = torch.tensor(np.random.random(t.shape) >= p, dtype=torch.float64)
    
    scale = 1 / (1 - p)
    output = t * mask * scale
    dropout_pattern = mask * scale
    
    return output.numpy(), dropout_pattern.numpy()