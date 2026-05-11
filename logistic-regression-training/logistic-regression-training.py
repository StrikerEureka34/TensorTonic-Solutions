import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        err = p - y
        w -= lr * (X.T @ err) / len(y)
        b -= lr * err.mean()
    return w.tolist(), float(b)
    pass