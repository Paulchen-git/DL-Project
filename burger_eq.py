# Burgers' equation solver using deep neural networks

import scipy
import torch
import numpy as np
from pyDOE import lhs

# Set random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

class PINN(torch.nn.Module):
    def __init__(self, layers, activation):
        super(PINN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.network = self.build_network()

    def build_network(self):
        modules = []
        for i in range(len(self.layers) - 1):
            modules.append(torch.nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                modules.append(self.activation)
        return torch.nn.Sequential(*modules)
    
    def forward(self, t, x):
        inputs = torch.cat((t, x), dim=1)
        return self.network(inputs)

def u(t, x):
    """    solution approximated with a neural network.
    """
    return model.forward(t, x)


def f(t, x):
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    u_ = u(t, x)
    u_t = torch.autograd.grad(u_, t, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_x = torch.autograd.grad(u_, x, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u_ * u_x - 0.01 / torch.pi * u_xx
    

def initial_condition(x):
    """Initial condition u(0, x) = -sin(pi*x)
    """
    return -torch.sin(torch.pi * x)

def boundary_condition(t):
    """Boundary condition u(t, -1) = u(t, 1) = 0
    """
    return torch.zeros_like(t)

def exact_solution(t, x):
    """Exact solution for comparison.
    """
    return -torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * 0.01 * t)

def loss_function(X_f_train, X_u_train):
    """Compute the total loss for the PINN.
    """
    # Physics loss
    f_pred = f(X_f_train[:, 0:1], X_f_train[:, 1:2])
    loss_f = torch.mean(f_pred**2)

    # Initial condition loss
    u_i_pred = u(X_u_train[:, 0:1], X_u_train[:, 1:2])
    u_i_exact = initial_condition(X_u_train[:, 1:2])
    loss_i = torch.mean((u_i_pred - u_i_exact)**2)

    # Boundary condition loss
    u_b_pred = u(X_u_train[:, 0:1], X_u_train[:, 1:2])
    u_b_exact = boundary_condition(X_u_train[:, 0:1])
    loss_b = torch.mean((u_b_pred - u_b_exact)**2)

    return loss_f + loss_i + loss_b

def train(model, optimizer, epochs, X_f_train, X_u_train):
    """Train the PINN model.
    """
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(X_f_train, X_u_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        loss_history.append(loss.item())

    return model, loss_history

def plot_solution(model, t, x):
    """Plot the solution u(t, x) over a grid.
    """
    import matplotlib.pyplot as plt

    T, X = np.meshgrid(t.numpy(), x.numpy())
    U = model.forward(torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1),
                      torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1))
    U = U.detach().numpy().reshape(T.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(T, X, U, levels=50, cmap='viridis')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Burgers\' Equation Solution')
    plt.savefig('burgers_solution.png')
    plt.close()

def plot_loss(loss_history):
    """Plot the training loss over epochs.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.savefig('loss_history.png')
    plt.close()

if __name__ == "__main__":
    # Define the neural network architecture 9 layers with 20 neurons in each hidden layer
    layers = [2] + [20]*8 + [1]
    activation = torch.nn.Tanh()
    model = PINN(layers, activation)

    # display nb of parameters
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {nb_params}')

    # Define optimizer
    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)

    N_u = 100  # Number of initial condition points
    N_f = 8000  # Number of collocation points

    data = scipy.io.loadmat('./burgers_shock.mat')
    
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
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]

    # Plot initial and boundary conditions and collocation points
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X_u_train[:, 1], X_u_train[:, 0], s=5, color='red', label='Initial/Boundary Condition Points')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Training Data Points')
    plt.legend()
    plt.savefig('training_data_points.png')
    plt.close()

    
    # Train the model
    trained_model, loss_history = train(model, optimizer, epochs=1000, X_f_train=torch.tensor(X_f_train, dtype=torch.float32), X_u_train=torch.tensor(X_u_train, dtype=torch.float32)) 

    # Plot the solution and initial/boundary conditions and collocation points
    t_plot = torch.linspace(0, 1, 100).unsqueeze(1)
    x_plot = torch.linspace(-1, 1, 100).unsqueeze(1)
    plot_solution(trained_model, t_plot, x_plot)
    # Plot the loss history
    plot_loss(loss_history)  # Uncomment and implement loss history tracking in training function

    #Calculate L2 error
    u_pred = trained_model.forward(torch.tensor(X_star[:,1:2], dtype=torch.float32), torch.tensor(X_star[:,0:1], dtype=torch.float32))
    u_pred = u_pred.detach().numpy()
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print(f'L2 relative error: {error_u}')

