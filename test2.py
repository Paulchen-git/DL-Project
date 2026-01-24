import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Set random seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


class PINN(nn.Module):
    """Physics-Informed Neural Network for solving Burgers equation."""
    
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = nn.Tanh()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activation(layer(inputs))
        
        output = self.layers[-1](inputs)
        return output


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


def generate_training_data(n_initial, n_boundary, n_collocation, x_range=(-1, 1), t_range=(0, 1)):
    """
    Generate training data for Burgers equation.
    
    Args:
        n_initial: Number of initial condition points
        n_boundary: Number of boundary points per boundary
        n_collocation: Number of interior collocation points
        x_range: Spatial domain range
        t_range: Temporal domain range
    """
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
    
    # Interior collocation points for PDE
    x_col = torch.rand(n_collocation, 1, requires_grad=True, device=device) * (x_max - x_min) + x_min
    t_col = torch.rand(n_collocation, 1, requires_grad=True, device=device) * (t_max - t_min) + t_min
    
    return (x_init, t_init, u_init, 
            x_bc_left, t_bc_left, u_bc_left,
            x_bc_right, t_bc_right, u_bc_right,
            x_col, t_col)


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


def evaluate_solution(model, n_x=200, n_t=200, x_range=(-1, 1), t_range=(0, 1)):
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


def visualize_results(X, T, U, loss_history):
    """Visualize training results and solutions."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Loss history - Total
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.semilogy(loss_history['total'], label='Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss history - Components
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.semilogy(loss_history['pde'], label='PDE', linewidth=2)
    ax2.semilogy(loss_history['ic'], label='Initial Condition', linewidth=2)
    ax2.semilogy(loss_history['bc'], label='Boundary Condition', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Solution (2D contour)
    ax3 = fig.add_subplot(3, 3, 3)
    c3 = ax3.contourf(X, T, U, levels=50, cmap='RdBu_r')
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_title('PINN Solution u(x,t)')
    plt.colorbar(c3, ax=ax3)
    
    # Solution (3D surface)
    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(X, T, U, cmap='RdBu_r', alpha=0.9, edgecolor='none')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_zlabel('u')
    ax4.set_title('Solution Surface')
    ax4.view_init(elev=25, azim=45)
    
    # Solution at different time snapshots
    ax5 = fig.add_subplot(3, 3, 5)
    time_indices = [0, U.shape[0]//4, U.shape[0]//2, 3*U.shape[0]//4, U.shape[0]-1]
    time_values = [T[idx, 0] for idx in time_indices]
    
    for idx, t_val in zip(time_indices, time_values):
        ax5.plot(X[idx, :], U[idx, :], label=f't={t_val:.2f}', linewidth=2)
    
    ax5.set_xlabel('x')
    ax5.set_ylabel('u')
    ax5.set_title('Solution Snapshots at Different Times')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Solution evolution at x=0
    ax6 = fig.add_subplot(3, 3, 6)
    mid_idx = U.shape[1] // 2
    ax6.plot(T[:, mid_idx], U[:, mid_idx], 'b-', linewidth=2)
    ax6.set_xlabel('t')
    ax6.set_ylabel('u')
    ax6.set_title('Solution Evolution at x=0')
    ax6.grid(True, alpha=0.3)
    
    # Heatmap with contour lines
    ax7 = fig.add_subplot(3, 3, 7)
    c7 = ax7.contourf(X, T, U, levels=50, cmap='RdBu_r')
    ax7.contour(X, T, U, levels=10, colors='black', linewidths=0.5, alpha=0.4)
    ax7.set_xlabel('x')
    ax7.set_ylabel('t')
    ax7.set_title('Solution with Contour Lines')
    plt.colorbar(c7, ax=ax7)
    
    # Initial condition verification
    ax8 = fig.add_subplot(3, 3, 8)
    x_ic = X[0, :]
    u_ic_pred = U[0, :]
    u_ic_exact = -np.sin(np.pi * x_ic)
    ax8.plot(x_ic, u_ic_pred, 'b-', label='PINN', linewidth=2)
    ax8.plot(x_ic, u_ic_exact, 'r--', label='Exact IC', linewidth=2)
    ax8.set_xlabel('x')
    ax8.set_ylabel('u')
    ax8.set_title('Initial Condition: t=0')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Solution statistics over time
    ax9 = fig.add_subplot(3, 3, 9)
    u_min = np.min(U, axis=1)
    u_max = np.max(U, axis=1)
    u_mean = np.mean(U, axis=1)
    t_vals = T[:, 0]
    
    ax9.plot(t_vals, u_min, label='Min', linewidth=2)
    ax9.plot(t_vals, u_max, label='Max', linewidth=2)
    ax9.plot(t_vals, u_mean, label='Mean', linewidth=2)
    ax9.fill_between(t_vals, u_min, u_max, alpha=0.3)
    ax9.set_xlabel('t')
    ax9.set_ylabel('u')
    ax9.set_title('Solution Statistics Over Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("pinn_burgers_results.png", dpi=300)


def main():
    """Main execution function."""
    
    # Hyperparameters
    nu = 0.01 / np.pi  # Viscosity coefficient
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]  # Network architecture
    n_initial = 50       # Number of initial condition points
    n_boundary = 50      # Number of boundary points per boundary
    n_collocation = 10000 # Number of interior collocation points
    epochs = 5000
    learning_rate = 0.001
    lambda_ic = 10   # Weight for initial condition loss
    lambda_bc = 10   # Weight for boundary condition loss
    
    print("=" * 70)
    print("Physics-Informed Neural Network for Burgers Equation")
    print("=" * 70)
    print(f"Domain: x ∈ [-1, 1], t ∈ [0, 1]")
    print(f"PDE: u_t + u*u_x - ν*u_xx = 0")
    print(f"Viscosity: ν = {nu:.6f} (0.01/π)")
    print(f"Initial condition: u(x, 0) = -sin(πx)")
    print(f"Boundary conditions: u(-1, t) = u(1, t) = 0")
    print(f"Network architecture: {layers}")
    print(f"Initial condition points: {n_initial}")
    print(f"Boundary points (per boundary): {n_boundary}")
    print(f"Interior collocation points: {n_collocation}")
    print("=" * 70)
    
    # Initialize model
    model = PINN(layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate training data
    print("\nGenerating training data...")
    data = generate_training_data(n_initial, n_boundary, n_collocation)
    
    # Train model
    print("\nTraining PINN...")
    loss_history = train_pinn(model, optimizer, data, nu, epochs, lambda_ic, lambda_bc, log_interval=100)
    
    # Evaluate solution
    print("\nEvaluating solution on grid...")
    X, T, U = evaluate_solution(model, n_x=200, n_t=200)
    
    # Check initial condition error
    u_init_pred = U[0, :]
    x_init = X[0, :]
    u_init_exact = -np.sin(np.pi * x_init)
    ic_error = np.mean(np.abs(u_init_pred - u_init_exact))
    print(f"\nInitial condition mean absolute error: {ic_error:.6e}")
    
    # Check boundary condition errors
    bc_left_error = np.mean(np.abs(U[:, 0]))
    bc_right_error = np.mean(np.abs(U[:, -1]))
    print(f"Left boundary condition mean absolute error: {bc_left_error:.6e}")
    print(f"Right boundary condition mean absolute error: {bc_right_error:.6e}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(X, T, U, loss_history)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()