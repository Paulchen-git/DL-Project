import torch
import numpy as np
from pyDOE import lhs
import scipy.io

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

def generate_data(N_u, N_f, device='cpu'):
    """Generate training data for Burgers equation using Latin Hypercube Sampling.
    Args:
    N_u: Number of initial and boundary condition points
    N_f: Number of collocation points
    """
    data = scipy.io.loadmat('./data/burgers_shock.mat')

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])
    
    # Sample N_u points from all IC/BC points
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # Generate collocation points using Latin Hypercube Sampling (without adding IC/BC points)
    X_f_train = lb + (ub-lb)*lhs(2, N_f)

    # Convert to tensors - explicitly on the specified device
    x_u = torch.tensor(X_u_train[:,0:1], dtype=torch.float32, device=device, requires_grad=False)
    t_u = torch.tensor(X_u_train[:,1:2], dtype=torch.float32, device=device, requires_grad=False)
    u = torch.tensor(u_train, dtype=torch.float32, device=device, requires_grad=False)
    
    x_col = torch.tensor(X_f_train[:,0:1], dtype=torch.float32, device=device, requires_grad=False)
    t_col = torch.tensor(X_f_train[:,1:2], dtype=torch.float32, device=device, requires_grad=False)

    return (x_u, t_u, u, x_col, t_col)
