from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy

def visualize_results(X, T, U, loss_history, path="pinn_burgers_results.png"):
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
    c3 = ax3.contourf(T, X, U, levels=50, cmap='RdBu_r')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
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
    
    # Exact solution
    ax7 = fig.add_subplot(3, 3, 7)
    data = scipy.io.loadmat('./burgers_shock.mat')
    
    t_exact = data['t'].flatten()[:,None]
    x_exact = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X_exact, T_exact = np.meshgrid(x_exact, t_exact)
    
    X_star = np.hstack((X_exact.flatten()[:,None], T_exact.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    c7 = ax7.tricontourf(T_exact.flatten(), X_exact.flatten(), u_star.flatten(), levels=50, cmap='RdBu_r')
    plt.colorbar(c7, ax=ax7, label='Exact Solution u(x,t)')
    ax7.set_xlabel('x')
    ax7.set_ylabel('t')
    ax7.set_title('Exact Solution u(x,t)')
    
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
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

def visualize_collocation_points(data):
    """Visualize collocation points used in training."""

    (x_init, t_init, u_init, 
            x_bc_left, t_bc_left, u_bc_left,
            x_bc_right, t_bc_right, u_bc_right,
            x_col, t_col, u_col) = data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, )
    # add exact solution heatmap in the background
    data = scipy.io.loadmat('./burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    c = ax.tricontourf(T.flatten(), X.flatten(), u_star.flatten(), levels=50, cmap='RdBu_r')
    plt.colorbar(c, ax=ax, label='Exact Solution u(x,t)')
    ax.scatter(t_init.cpu(), x_init.cpu(), c='red', label='Initial Condition', s=20)
    ax.scatter(t_bc_left.cpu(), x_bc_left.cpu(), c='green', label='Boundary Condition Left', s=20)
    ax.scatter(t_bc_right.cpu(), x_bc_right.cpu(), c='blue', label='Boundary Condition Right', s=20)
    ax.scatter(t_col.cpu().detach(), x_col.cpu().detach(), c='black', label='Collocation Points', s=5, marker='x')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    
    ax.set_title('Collocation Points for PINN Training')
    ax.set_aspect(0.25)
    ax.legend()
    plt.savefig("pinn_burgers_collocation_points.png", dpi=300)
    plt.show()
    plt.close()
    