import numpy as np

import torch

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
        return torch.empty(0, max_len or 0)
    
    tensors = [torch.as_tensor(s) for s in seqs]
    target_len = max_len if max_len is not None else max(len(t) for t in tensors)
    
    dtype = tensors[0].dtype if tensors and len(tensors[0]) > 0 else torch.long
    out = torch.full((len(seqs), target_len), pad_value, dtype=dtype)
    
    for i, t in enumerate(tensors):
        length = min(len(t), target_len)
        if length > 0:
            out[i, :length] = t[:length]
            
    return out