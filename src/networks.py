import torch
import torch.nn as nn
from .burger_eq import compute_pde_residual
from .poisson_eq import compute_pde_residual as compute_pde_residual_poisson
from tqdm import tqdm


class PINNforward(nn.Module):
    """Physics-Informed Neural Network for forward problem solving Burgers equation."""

    def __init__(self, layers, activation):
        super(PINNforward, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        self.activation = activation
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
    
    def train_pinn(self, optimizer, data, nu, epochs, lambda_pde=1, lambda_icbc=1, log_interval=100):
        """
        Train the PINN model.
        
        Args:
            optimizer: Optimizer (Adam, L-BFGS, etc.)
            data: Tuple of training data
            nu: Viscosity coefficient
            epochs: Number of training epochs
            lambda_ic: Weight for initial condition loss
            lambda_bc: Weight for boundary condition loss
            log_interval: Interval for logging training progress
        """
        (x_u, t_u, u, x_col, t_col) = data
        
        loss_history = {'total': [], 'pde': [], 'ib': []}
        
        # Check if optimizer is L-BFGS
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        
        # Dictionary to store current loss values for L-BFGS
        current_losses = {'total': [], 'pde': [], 'ib': []}
        
        def closure():
            """Closure function required by L-BFGS optimizer"""
            optimizer.zero_grad()
            
            # PDE residual loss (interior points)
            residual = compute_pde_residual(self, x_col, t_col, nu)
            loss_pde = torch.mean(residual**2)
            
            # Initial and boundary condition loss
            u_pred_ib = self.forward(x_u, t_u)
            loss_ib = torch.mean((u_pred_ib - u)**2)
            
            
            
            # Total loss
            loss = lambda_pde * loss_pde + lambda_icbc * loss_ib
            
            # Store current losses for logging (detach to avoid keeping computation graph)
            current_losses['total'].append(loss.item())
            current_losses['pde'].append(loss_pde.item())
            current_losses['ib'].append(loss_ib.item())
            
            loss.backward()
            return loss
        
        for epoch in tqdm(range(epochs), desc="Training PINN"):
            if is_lbfgs:
                # L-BFGS requires a closure function
                optimizer.step(closure)
                
                # Use stored loss values from closure
                loss = current_losses['total'][-1]
                loss_pde = current_losses['pde'][-1]
                loss_ib = current_losses['ib'][-1]
            else:
                # Standard optimizer (Adam, SGD, etc.)
                optimizer.zero_grad()
                
                # PDE residual loss (interior points)
                residual = compute_pde_residual(self, x_col, t_col, nu)
                loss_pde = torch.mean(residual**2)
                
                # Initial condition loss
                u_pred_ib = self.forward(x_u, t_u)
                loss_ib = torch.mean((u_pred_ib - u)**2)
                
                # Total loss
                loss = lambda_pde * loss_pde + lambda_icbc * loss_ib
                loss.backward()
                optimizer.step()
                
                # Convert to items for storage
                loss = loss.item()
                loss_pde = loss_pde.item()
                loss_ib = loss_ib.item()
            
            loss_history['total'].append(loss)
            loss_history['pde'].append(loss_pde)
            loss_history['ib'].append(loss_ib)
            
            if log_interval is not None:
                if (epoch + 1) % log_interval == 0:
                    print(f"Epoch {epoch+1}/{epochs} \
                        -- Total Loss: {loss:.6f} \
                        -- PDE Loss: {loss_pde:.6f} \
                        -- IC Loss: {loss_ib:.6f}")
            
        return loss_history, current_losses
    
    def save_model(self, path):
        """Save the model state dictionary to the specified path."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path, device='cpu'):
        """Load the model state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)


class PINNinverse(nn.Module):
    """
    Physics-Informed Neural Network for inverse problem (parameter identification).
    
    This class learns both the solution u(x,t) and the PDE parameters:
    - lambda_1: coefficient of the nonlinear term u*u_x
    - lambda_2: viscosity coefficient (nu)
    
    The Burgers equation is: u_t + lambda_1*u*u_x - lambda_2*u_xx = 0
    """

    def __init__(self, layers, activation, lb, ub):
        """
        Initialize the PINNinverse model.
        
        Args:
            layers: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function (e.g., nn.Tanh())
            lb: Lower bounds for input normalization [x_min, t_min]
            ub: Upper bounds for input normalization [x_max, t_max]
        """
        super(PINNinverse, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Store bounds for input normalization
        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float32))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float32))
        
        # Initialize learnable PDE parameters
        # lambda_1: coefficient for u*u_x (expected value ~1.0 for Burgers)
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # lambda_2: log of viscosity coefficient (for numerical stability)
        # exp(lambda_2) should be close to nu = 0.01/pi ≈ 0.00318
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], dtype=torch.float32))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        """
        Forward pass with input normalization.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            u: Predicted solution
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        
        # Normalize inputs to [-1, 1]
        inputs = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        
        # Pass through network
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activation(layer(inputs))
        
        output = self.layers[-1](inputs)
        return output
    
    def compute_pde_residual(self, x, t):
        """
        Compute PDE residual using learned parameters.
        
        The residual is: u_t + lambda_1*u*u_x - lambda_2*u_xx
        where lambda_2 = exp(self.lambda_2) for numerical stability
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            residual: PDE residual
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        
        # Burgers equation with learned parameters
        # lambda_2 is stored in log-space for numerical stability
        lambda_2 = torch.exp(self.lambda_2)
        residual = u_t + self.lambda_1 * u * u_x - lambda_2 * u_xx
        
        return residual
    
    def train_inverse(self, optimizer, x_data, t_data, u_data, x_col, t_col, 
                     epochs, log_interval=100):
        """
        Train the inverse PINN to learn both the solution and PDE parameters.
        
        Args:
            optimizer: PyTorch optimizer
            x_data: Spatial coordinates of data points
            t_data: Temporal coordinates of data points
            u_data: Solution values at data points
            x_col: Spatial coordinates of collocation points
            t_col: Temporal coordinates of collocation points
            epochs: Number of training epochs
            log_interval: Frequency of logging (None to disable)
            
        Returns:
            loss_history: Dictionary containing loss history
        """
        loss_history = {
            'total': [], 
            'data': [], 
            'pde': [],
            'lambda_1': [],
            'lambda_2': []
        }
        
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        current_losses = {'total': [], 'data': [], 'pde': [], 'lambda_1': [], 'lambda_2': []}
        
        def closure():
            """Closure function for L-BFGS optimizer."""
            optimizer.zero_grad()
            
            # Data fitting loss
            u_pred = self.forward(x_data, t_data)
            loss_data = torch.mean((u_pred - u_data)**2)
            
            # PDE residual loss
            residual = self.compute_pde_residual(x_col, t_col)
            loss_pde = torch.mean(residual**2)
            
            # Total loss
            loss = loss_data + loss_pde
            
            # Store for logging
            current_losses['total'].append(loss.item())
            current_losses['data'].append(loss_data.item())
            current_losses['pde'].append(loss_pde.item())
            current_losses['lambda_1'].append(self.lambda_1.item())
            current_losses['lambda_2'].append(torch.exp(self.lambda_2).item())
            
            loss.backward()
            return loss
        
        for epoch in tqdm(range(epochs), desc="Training Inverse PINN"):
            if is_lbfgs:
                optimizer.step(closure)
                loss = current_losses['total'][-1]
                loss_data = current_losses['data'][-1]
                loss_pde = current_losses['pde'][-1]
            else:
                optimizer.zero_grad()
                
                # Data fitting loss
                u_pred = self.forward(x_data, t_data)
                loss_data = torch.mean((u_pred - u_data)**2)
                
                # PDE residual loss
                residual = self.compute_pde_residual(x_col, t_col)
                loss_pde = torch.mean(residual**2)
                
                # Total loss
                loss = loss_data + loss_pde
                loss.backward()
                optimizer.step()
                
                loss = loss.item()
                loss_data = loss_data.item()
                loss_pde = loss_pde.item()
            
            # Store history
            loss_history['total'].append(loss)
            loss_history['data'].append(loss_data)
            loss_history['pde'].append(loss_pde)
            loss_history['lambda_1'].append(self.lambda_1.item())
            loss_history['lambda_2'].append(torch.exp(self.lambda_2).item())
            
            # Logging
            if log_interval is not None and (epoch + 1) % log_interval == 0:
                lambda_2_val = torch.exp(self.lambda_2).item()
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Total Loss: {loss:.6e}")
                print(f"  Data Loss: {loss_data:.6e}")
                print(f"  PDE Loss: {loss_pde:.6e}")
                print(f"  λ₁: {self.lambda_1.item():.5f}")
                print(f"  λ₂ (ν): {lambda_2_val:.7f}")
        
        return loss_history, current_losses
    
    def get_parameters(self):
        """
        Get the identified PDE parameters.
        
        Returns:
            Dictionary with lambda_1 and lambda_2 (nu) values
        """
        return {
            'lambda_1': self.lambda_1.item(),
            'lambda_2': torch.exp(self.lambda_2).item()
        }
    
    def save_model(self, path):
        """Save model and parameters."""
        state = {
            'state_dict': self.state_dict(),
            'lambda_1': self.lambda_1.item(),
            'lambda_2': torch.exp(self.lambda_2).item(),
            'lb': self.lb.cpu().numpy(),
            'ub': self.ub.cpu().numpy()
        }
        torch.save(state, path)