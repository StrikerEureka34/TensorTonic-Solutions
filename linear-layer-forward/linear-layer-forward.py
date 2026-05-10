def linear_layer_forward(X, W, b):
    import torch
    X = torch.tensor(X, dtype=torch.float32)
    W = torch.tensor(W, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    return (X @ W + b).tolist()