def xavier_initialization(W, fan_in, fan_out):
    import torch
    W = torch.tensor(W, dtype=torch.float32)
    L = (6 / (fan_in + fan_out)) ** 0.5
    return W * 2 * L - L