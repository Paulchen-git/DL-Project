import torch
import numpy as np

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
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
    
    def forward(self, x, y):
        inputs = torch.cat((x, y), dim=1)
        return self.network(inputs)
    
def u(x, y, model):
    """Solution approximated with a neural network.
    """
    return model.forward(x, y)

def f(x, y, model):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    u_ = u(x, y, model)
    u_xx = torch.autograd.grad(torch.autograd.grad(u_, x, grad_outputs=torch.ones_like(u_), create_graph=True)[0], x, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    u_yy = torch.autograd.grad(torch.autograd.grad(u_, y, grad_outputs=torch.ones_like(u_), create_graph=True)[0], y, grad_outputs=torch.ones_like(u_), create_graph=True)[0]
    return u_xx + u_yy + 2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def boundary_condition(x, y):
    """Boundary condition u(x, y) = 0 on the boundary of the domain.
    """
    return torch.zeros_like(x)

def exact_solution(x, y):
    """Exact solution for comparison.
    """
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def loss(pred_u, true_u, pred_f):
    """Compute the total loss as the sum of the MSE of u and f.
    """
    mse_u = torch.mean((pred_u - true_u) ** 2)
    mse_f = torch.mean(pred_f ** 2)
    return mse_u + mse_f

def train(model, optimizer, X_f_train, X_u_train, u_train, epochs):
    """Train the PINN model.
    """
    X_f_train = X_f_train.to(device)
    X_u_train = X_u_train.to(device)
    model.to(device)
    u_train = u_train.to(device)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Predict u and f
        u_pred = u(X_u_train[:, 0:1], X_u_train[:, 1:2], model)
        f_pred = f(X_f_train[:, 0:1], X_f_train[:, 1:2], model)
        
        # Compute loss
        loss_value = loss(u_pred, u_train, f_pred)
        
        # Backpropagation
        loss_value.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value.item()}')
        loss_history.append(loss_value.item())
    
    return model, loss_history

def plot_solution(model, X_star, u_star):
    """Plot the predicted solution against the exact solution in 2D.
    """
    import matplotlib.pyplot as plt

    u_pred = model.forward(torch.tensor(X_star[:,0:1], dtype=torch.float32), torch.tensor(X_star[:,1:2], dtype=torch.float32))
    u_pred = u_pred.detach().numpy()
    U_pred = u_pred.reshape(100, 100)
    U_exact = u_star.reshape(100, 100)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, U_pred, shading='auto')
    plt.colorbar()
    plt.title('Predicted Solution u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, U_exact, shading='auto')
    plt.colorbar()
    plt.title('Exact Solution u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('poisson_solution.png')
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

    layers = [2, 16, 32, 64, 64, 32, 16, 1]
    activation = torch.nn.Tanh()
    model = PINN(layers, activation)

    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.001)

    N_u = 200  # Number of boundary condition points
    N_f = 10000  # Number of collocation points

    # Generate training data
    # Boundary points
    x_u = np.random.rand(N_u, 1)
    y_u = np.random.rand(N_u, 1)
    u_u = boundary_condition(torch.tensor(x_u, dtype=torch.float32), torch.tensor(y_u, dtype=torch.float32))
    X_u_train = torch.tensor(np.hstack((x_u, y_u)), dtype=torch.float32)
    u_train = u_u
    # Collocation points
    x_f = np.random.rand(N_f, 1)
    y_f = np.random.rand(N_f, 1)
    X_f_train = torch.tensor(np.hstack((x_f, y_f)), dtype=torch.float32)

    # Train the model
    trained_model, loss_history = train(model, optimizer, X_f_train, X_u_train, u_train, epochs=5000)
    # Plot the solution
    # Create a grid for plotting
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    u_star = exact_solution(torch.tensor(X_star[:,0:1], dtype=torch.float32), torch.tensor(X_star[:,1:2], dtype=torch.float32)).detach().numpy()
    plot_solution(trained_model, X_star, u_star)
    # Plot the loss history
    plot_loss(loss_history) 


