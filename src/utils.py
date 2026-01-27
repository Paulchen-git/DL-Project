import scipy
import torch
from .burger_eq import compute_pde_residual
import numpy as np

def evaluate_solution(model, n_x=200, n_t=200, x_range=(-1, 1), t_range=(0, 1), device='cpu'):
    """Evaluate solution on a grid."""
    
    x = np.linspace(x_range[0], x_range[1], n_x)
    t = np.linspace(t_range[0], t_range[1], n_t)
    X, T = np.meshgrid(x, t)
    
    # Create tensors on the same device as the model
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device, requires_grad=False)
    t_test = torch.tensor(T.flatten()[:, None], dtype=torch.float32, device=device, requires_grad=False)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test)
        # Transfer to CPU only if needed (MPS -> CPU for numpy conversion)
        if u_pred.device.type != 'cpu':
            u_pred = u_pred.cpu()
        u_pred = u_pred.numpy()
    
    U = u_pred.reshape(n_t, n_x)
    
    return X, T, U

def approximation_error(model):
    """Compute absolute and relative error between predicted and exact solutions."""
    data = scipy.io.loadmat('./data/burgers_shock.mat')
    
    t_exact = data['t'].flatten()[:,None]
    x_exact = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X_exact, T_exact = np.meshgrid(x_exact, t_exact)
    
    X_star = np.hstack((X_exact.flatten()[:,None], T_exact.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]

    # Get the device from the model's parameters
    device = next(model.parameters()).device
    
    # Create tensors on the same device as the model
    x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device, requires_grad=False)
    t_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device, requires_grad=False)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test)
        # Transfer to CPU only if on a different device
        if u_pred.device.type != 'cpu':
            u_pred = u_pred.cpu()
        u_pred = u_pred.numpy()
    
    error = np.abs(u_pred - u_star)
    relative_error = np.linalg.norm(error) / np.linalg.norm(u_star)
    
    return error, relative_error
