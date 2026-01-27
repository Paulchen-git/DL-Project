from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy

def visualize_results(X, T, U, loss_history, path="pinn_burgers_results.png"):
    """Visualize training results and solutions."""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Loss history - Total
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.semilogy(loss_history['total'], label='Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss history - Components
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(loss_history['pde'], label='PDE', linewidth=2)
    ax2.semilogy(loss_history['ib'], label='IC/BC Condition', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Solution (2D contour)
    ax3 = fig.add_subplot(2, 3, 4)
    c3 = ax3.contourf(T, X, U, levels=50, cmap='RdBu_r')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_title('PINN Solution u(x,t)')
    plt.colorbar(c3, ax=ax3)
    
    # Solution at different time snapshots
    ax4 = fig.add_subplot(2, 3, 6)
    time_indices = [0, U.shape[0]//4, U.shape[0]//2, 3*U.shape[0]//4, U.shape[0]-1]
    time_values = [T[idx, 0] for idx in time_indices]
    
    for idx, t_val in zip(time_indices, time_values):
        ax4.plot(X[idx, :], U[idx, :], label=f't={t_val:.2f}', linewidth=2)
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('u')
    ax4.set_title('Solution Snapshots at Different Times')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Exact solution
    ax5 = fig.add_subplot(2, 3, 5)
    data = scipy.io.loadmat('./data/burgers_shock.mat')
    
    t_exact = data['t'].flatten()[:,None]
    x_exact = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X_exact, T_exact = np.meshgrid(x_exact, t_exact)
    
    X_star = np.hstack((X_exact.flatten()[:,None], T_exact.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    c5 = ax5.tricontourf(T_exact.flatten(), X_exact.flatten(), u_star.flatten(), levels=50, cmap='RdBu_r')
    plt.colorbar(c5, ax=ax5, label='Exact Solution u(x,t)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('t')
    ax5.set_title('Exact Solution u(x,t)')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

def visualize_collocation_points(data):
    """Visualize collocation points used in training."""

    (x_u, t_u, u, x_col, t_col) = data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, )
    # add exact solution heatmap in the background
    data = scipy.io.loadmat('./data/burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    c = ax.tricontourf(T.flatten(), X.flatten(), u_star.flatten(), levels=50, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, label='Exact Solution u(x,t)', shrink=0.5)
    # Transfer tensors to CPU for matplotlib visualization
    t_u_cpu = t_u.cpu() if t_u.device.type != 'cpu' else t_u
    x_u_cpu = x_u.cpu() if x_u.device.type != 'cpu' else x_u
    t_col_cpu = t_col.detach().cpu() if t_col.device.type != 'cpu' else t_col.detach()
    x_col_cpu = x_col.detach().cpu() if x_col.device.type != 'cpu' else x_col.detach()
    
    ax.scatter(t_u_cpu, x_u_cpu, c='red', label='Initial and Boundary Condition', s=20)
    ax.scatter(t_col_cpu, x_col_cpu, c='black', label='Collocation Points', s=5, marker='x')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    
    ax.set_title('Collocation Points for PINN Training')
    ax.set_aspect(0.25)
    ax.legend()
    plt.savefig("pinn_burgers_collocation_points.png", dpi=300)
    plt.show()
    plt.close()
    