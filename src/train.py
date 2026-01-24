import scipy
import torch
from .burger_eq import compute_pde_residual
import numpy as np

def train_pinn(model, optimizer, data, nu, epochs, lambda_ic=10, lambda_bc=10, log_interval=100):
    """
    Train the PINN model.
    
    Args:
        model: PINN model
        optimizer: Optimizer
        data: Tuple of training data
        nu: Viscosity coefficient
        epochs: Number of training epochs
        lambda_ic: Weight for initial condition loss
        lambda_bc: Weight for boundary condition loss
        log_interval: Interval for logging training progress
    """
    (x_init, t_init, u_init,
     x_bc_left, t_bc_left, u_bc_left,
     x_bc_right, t_bc_right, u_bc_right,
     x_col, t_col) = data
    
    loss_history = {'total': [], 'pde': [], 'ic': [], 'bc': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE residual loss (interior points)
        residual = compute_pde_residual(model, x_col, t_col, nu)
        loss_pde = torch.mean(residual**2)
        
        # Initial condition loss
        u_pred_init = model(x_init, t_init)
        loss_ic = torch.mean((u_pred_init - u_init)**2)
        
        # Boundary condition loss (left)
        u_pred_bc_left = model(x_bc_left, t_bc_left)
        loss_bc_left = torch.mean((u_pred_bc_left - u_bc_left)**2)
        
        # Boundary condition loss (right)
        u_pred_bc_right = model(x_bc_right, t_bc_right)
        loss_bc_right = torch.mean((u_pred_bc_right - u_bc_right)**2)
        
        loss_bc = loss_bc_left + loss_bc_right
        
        # Total loss
        loss = loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
        
        loss.backward(retain_graph=True)
        optimizer.step()
        
        loss_history['total'].append(loss.item())
        loss_history['pde'].append(loss_pde.item())
        loss_history['ic'].append(loss_ic.item())
        loss_history['bc'].append(loss_bc.item())
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {loss.item():.6f}")
            print(f"  PDE Loss: {loss_pde.item():.6f}")
            print(f"  IC Loss: {loss_ic.item():.6f}")
            print(f"  BC Loss: {loss_bc.item():.6f}")
    
    return loss_history

def evaluate_solution(model, n_x=200, n_t=200, x_range=(-1, 1), t_range=(0, 1), device='cpu'):
    """Evaluate solution on a grid."""
    
    x = np.linspace(x_range[0], x_range[1], n_x)
    t = np.linspace(t_range[0], t_range[1], n_t)
    X, T = np.meshgrid(x, t)
    
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    t_test = torch.tensor(T.flatten()[:, None], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy()
    
    U = u_pred.reshape(n_t, n_x)
    
    return X, T, U

def approximation_error(model):
    """Compute absolute and relative error between predicted and exact solutions."""
    data = scipy.io.loadmat('./burgers_shock.mat')
    
    t_exact = data['t'].flatten()[:,None]
    x_exact = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X_exact, T_exact = np.meshgrid(x_exact, t_exact)
    
    X_star = np.hstack((X_exact.flatten()[:,None], T_exact.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]

    x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
    t_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy()
    error = np.abs(u_pred - u_star)
    relative_error = np.linalg.norm(error) / np.linalg.norm(u_star)
    
    return error, relative_error