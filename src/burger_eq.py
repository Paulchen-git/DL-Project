import torch
import numpy as np
from pyDOE import lhs

def compute_pde_residual(model, x, t, nu):
    """
    Compute PDE residual for Burgers equation: u_t + u*u_x - nu*u_xx = 0
    
    Args:
        model: PINN model
        x: spatial coordinate
        t: temporal coordinate
        nu: viscosity coefficient
    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    
    # Second derivative with respect to x
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    
    # Burgers equation: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    
    return residual

def initial_condition(x):
    """Initial condition: u(x, 0) = -sin(πx)"""
    return -torch.sin(np.pi * x)


def boundary_condition_left(t):
    """Boundary condition at x = -1: u(-1, t) = 0"""
    return torch.zeros_like(t)


def boundary_condition_right(t):
    """Boundary condition at x = 1: u(1, t) = 0"""
    return torch.zeros_like(t)

def generate_training_data(n_initial, n_boundary, n_collocation, x_range=(-1, 1), t_range=(0, 1), device='cpu'):
    """
    Generate training data for Burgers equation.
    
    Args:
        n_initial: Number of initial condition points
        n_boundary: Number of boundary points per boundary
        n_collocation: Number of interior collocation points
        x_range: Spatial domain range
        t_range: Temporal domain range
    """
    import scipy.io
    
    x_min, x_max = x_range
    t_min, t_max = t_range
    
    # Initial condition points (t=0)
    x_init = torch.rand(n_initial, 1, device=device) * (x_max - x_min) + x_min
    t_init = torch.zeros(n_initial, 1, device=device)
    u_init = initial_condition(x_init)
    
    # Boundary condition points
    # Left boundary (x = x_min)
    t_bc_left = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min
    x_bc_left = torch.ones(n_boundary, 1, device=device) * x_min
    u_bc_left = boundary_condition_left(t_bc_left)
    
    # Right boundary (x = x_max)
    t_bc_right = torch.rand(n_boundary, 1, device=device) * (t_max - t_min) + t_min
    x_bc_right = torch.ones(n_boundary, 1, device=device) * x_max
    u_bc_right = boundary_condition_right(t_bc_right)
    
    # Interior collocation points for PDE using Latin Hypercube Sampling
    lhs_samples = lhs(2, n_collocation)
    x_col = torch.tensor(lhs_samples[:, 0:1], dtype=torch.float32, device=device) * (x_max - x_min) + x_min
    t_col = torch.tensor(lhs_samples[:, 1:2], dtype=torch.float32, device=device) * (t_max - t_min) + t_min

    # Load exact solution for supervised learning at collocation points
    data = scipy.io.loadmat('./burgers_shock.mat')
    t_exact = data['t'].flatten()
    x_exact = data['x'].flatten()
    Exact = np.real(data['usol']).T
    
    # Interpolate exact solution at collocation points
    from scipy.interpolate import griddata
    X_exact, T_exact = np.meshgrid(x_exact, t_exact)
    points_exact = np.hstack([X_exact.flatten()[:, None], T_exact.flatten()[:, None]])
    u_exact_flat = Exact.flatten()
    
    points_col = np.hstack([x_col.cpu().numpy(), t_col.cpu().numpy()])
    # Use linear interpolation (more stable than cubic) with nearest neighbor fallback
    u_col = griddata(points_exact, u_exact_flat, points_col, method='linear', fill_value=np.nan)
    
    # Fill any remaining NaN values with nearest neighbor interpolation
    nan_mask = np.isnan(u_col)
    if np.any(nan_mask):
        u_col_nearest = griddata(points_exact, u_exact_flat, points_col, method='nearest')
        u_col[nan_mask] = u_col_nearest[nan_mask]
    
    u_col = torch.tensor(u_col, dtype=torch.float32, device=device).reshape(-1, 1)
    
    return (x_init, t_init, u_init, 
            x_bc_left, t_bc_left, u_bc_left,
            x_bc_right, t_bc_right, u_bc_right,
            x_col, t_col, u_col)