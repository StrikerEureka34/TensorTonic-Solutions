def ridge_regression(X, y, lam):
    import torch
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    I = torch.eye(X.shape[1])
    w = torch.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    return w.tolist()