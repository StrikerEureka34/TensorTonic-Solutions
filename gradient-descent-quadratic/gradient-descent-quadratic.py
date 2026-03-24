import torch

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = torch.tensor([float(x0)], requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        loss = a * (x ** 2) + b * x + c
        loss.backward()
        optimizer.step()
        
    return x.item()