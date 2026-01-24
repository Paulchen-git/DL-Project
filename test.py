import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PINN(nn.Module):
    """Physics-Informed Neural Network for solving Poisson equation."""
    
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
    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activation(layer(inputs))
        
        output = self.layers[-1](inputs)
        return output


def f_source(x, y):
    """Source term: f(x,y) = 2π²sin(πx)sin(πy)"""
    return 2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)


def analytical_solution(x, y):
    """Analytical solution: u(x,y) = sin(πx)sin(πy)"""
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


def compute_pde_residual(model, x, y):
    """Compute PDE residual: ∇²u + f = 0"""
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    u = model(x, y)
    
    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0]
    
    # Poisson equation: ∇²u = -f
    laplacian = u_xx + u_yy
    f = f_source(x, y)
    residual = laplacian + f
    
    return residual


def generate_training_data(n_interior, n_boundary):
    """Generate collocation points for interior and boundary."""
    
    # Interior points
    x_int = torch.rand(n_interior, 1, requires_grad=True, device=device)
    y_int = torch.rand(n_interior, 1, requires_grad=True, device=device)
    
    # Boundary points (4 edges of unit square)
    n_per_edge = n_boundary // 4
    
    # Bottom: y=0
    x_b1 = torch.rand(n_per_edge, 1, device=device)
    y_b1 = torch.zeros(n_per_edge, 1, device=device)
    
    # Top: y=1
    x_b2 = torch.rand(n_per_edge, 1, device=device)
    y_b2 = torch.ones(n_per_edge, 1, device=device)
    
    # Left: x=0
    x_b3 = torch.zeros(n_per_edge, 1, device=device)
    y_b3 = torch.rand(n_per_edge, 1, device=device)
    
    # Right: x=1
    x_b4 = torch.ones(n_per_edge, 1, device=device)
    y_b4 = torch.rand(n_per_edge, 1, device=device)
    
    x_bc = torch.cat([x_b1, x_b2, x_b3, x_b4], dim=0)
    y_bc = torch.cat([y_b1, y_b2, y_b3, y_b4], dim=0)
    u_bc = torch.zeros(n_boundary, 1, device=device)  # Dirichlet BC: u=0
    
    return x_int, y_int, x_bc, y_bc, u_bc


def train_pinn(model, optimizer, x_int, y_int, x_bc, y_bc, u_bc, epochs, lambda_bc=10):
    """Train the PINN model."""
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE residual loss (interior points)
        residual = compute_pde_residual(model, x_int, y_int)
        loss_pde = torch.mean(residual**2)
        
        # Boundary condition loss
        u_pred_bc = model(x_bc, y_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        # Total loss
        loss = loss_pde + lambda_bc * loss_bc
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, "
                  f"PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}")
    
    return loss_history


def evaluate_solution(model, n_points=100):
    """Evaluate solution on a grid and compute error."""
    
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    y_test = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_pred = model(x_test, y_test).cpu().numpy()
        u_exact = analytical_solution(x_test, y_test).cpu().numpy()
    
    U_pred = u_pred.reshape(n_points, n_points)
    U_exact = u_exact.reshape(n_points, n_points)
    
    error = np.abs(U_pred - U_exact)
    relative_error = np.linalg.norm(error) / np.linalg.norm(U_exact)
    
    return X, Y, U_pred, U_exact, error, relative_error


def visualize_results(X, Y, U_pred, U_exact, error, loss_history):
    """Visualize training results and solutions."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Loss history
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.semilogy(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.grid(True, alpha=0.3)
    
    # PINN solution (2D)
    ax2 = fig.add_subplot(3, 3, 2)
    c2 = ax2.contourf(X, Y, U_pred, levels=50, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('PINN Solution')
    plt.colorbar(c2, ax=ax2)
    
    # Analytical solution (2D)
    ax3 = fig.add_subplot(3, 3, 3)
    c3 = ax3.contourf(X, Y, U_exact, levels=50, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Analytical Solution')
    plt.colorbar(c3, ax=ax3)
    
    # Absolute error (2D)
    ax4 = fig.add_subplot(3, 3, 4)
    c4 = ax4.contourf(X, Y, error, levels=50, cmap='hot')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Absolute Error')
    plt.colorbar(c4, ax=ax4)
    
    # PINN solution (3D)
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    surf5 = ax5.plot_surface(X, Y, U_pred, cmap='viridis', alpha=0.8)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('u')
    ax5.set_title('PINN Solution (3D)')
    
    # Analytical solution (3D)
    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    surf6 = ax6.plot_surface(X, Y, U_exact, cmap='viridis', alpha=0.8)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('u')
    ax6.set_title('Analytical Solution (3D)')
    
    # Error (3D)
    ax7 = fig.add_subplot(3, 3, 7, projection='3d')
    surf7 = ax7.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('error')
    ax7.set_title('Absolute Error (3D)')
    
    # Cross-section at y=0.5
    ax8 = fig.add_subplot(3, 3, 8)
    mid_idx = U_pred.shape[0] // 2
    ax8.plot(X[mid_idx, :], U_pred[mid_idx, :], 'b-', label='PINN', linewidth=2)
    ax8.plot(X[mid_idx, :], U_exact[mid_idx, :], 'r--', label='Analytical', linewidth=2)
    ax8.set_xlabel('x')
    ax8.set_ylabel('u')
    ax8.set_title('Cross-section at y=0.5')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Error distribution histogram
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.hist(error.flatten(), bins=50, color='red', alpha=0.7)
    ax9.set_xlabel('Absolute Error')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Error Distribution')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_poisson_results.png')
    plt.close()

def main():
    """Main execution function."""
    
    start_time = time.time()
    
    # Hyperparameters
    layers = [2, 32, 32, 32, 1]  # Network architecture
    n_interior = 20  # Number of interior collocation points
    n_boundary = 4   # Number of boundary collocation points
    epochs = 5000
    learning_rate = 0.001
    lambda_bc = 10  # Weight for boundary condition loss
    
    print("=" * 60)
    print("Physics-Informed Neural Network for Poisson Equation")
    print("=" * 60)
    print(f"Domain: [0,1] × [0,1]")
    print(f"PDE: ∇²u = -f(x,y), where f(x,y) = 2π²sin(πx)sin(πy)")
    print(f"BC: u = 0 on boundary")
    print(f"Network architecture: {layers}")
    print(f"Interior points: {n_interior}")
    print(f"Boundary points: {n_boundary}")
    print("=" * 60)
    
    # Initialize model
    model = PINN(layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate training data
    x_int, y_int, x_bc, y_bc, u_bc = generate_training_data(n_interior, n_boundary)
    
    # Train model
    print("\nTraining PINN...")
    loss_history = train_pinn(model, optimizer, x_int, y_int, x_bc, y_bc, u_bc, epochs, lambda_bc)
    
    # Evaluate solution
    print("\nEvaluating solution...")
    X, Y, U_pred, U_exact, error, relative_error = evaluate_solution(model, n_points=100)
    
    print(f"\nRelative L2 error: {relative_error:.6e}")
    print(f"Maximum absolute error: {np.max(error):.6e}")
    print(f"Mean absolute error: {np.mean(error):.6e}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(X, Y, U_pred, U_exact, error, loss_history)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()