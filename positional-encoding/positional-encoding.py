import numpy as np
import torch

def positional_encoding(seq_len, d_model, base=10000.0):
    pos = torch.arange(seq_len, dtype=torch.float64).unsqueeze(1)
    
    i = torch.arange(d_model // 2, dtype=torch.float64)
    div_term = torch.pow(base, 2 * i / d_model)
    
    i_sin = torch.arange((d_model + 1) // 2, dtype=torch.float64)
    div_term_sin = torch.pow(base, 2 * i_sin / d_model)
    
    pe = torch.zeros(seq_len, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos / div_term_sin)
    pe[:, 1::2] = torch.cos(pos / div_term)
    
    return pe.numpy()