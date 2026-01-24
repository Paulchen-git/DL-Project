import torch
import torch.nn as nn
from .burger_eq import compute_pde_residual


class PINN(nn.Module):
    """Physics-Informed Neural Network"""

    def __init__(self, layers, activation):
        super(PINN, self).__init__()
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
    
    def train_pinn(self, optimizer, data, nu, epochs, lambda_ic=10, lambda_bc=10, log_interval=100):
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
        x_col, t_col, u_col) = data
        
        loss_history = {'total': [], 'pde': [], 'ic': [], 'bc': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # PDE residual loss (interior points)
            residual = compute_pde_residual(self, x_col, t_col, nu)
            loss_pde = torch.mean(residual**2)
            
            # Initial condition loss
            u_pred_init = self.forward(x_init, t_init)
            loss_ic = torch.mean((u_pred_init - u_init)**2)
            
            # Boundary condition loss (left)
            u_pred_bc_left = self.forward(x_bc_left, t_bc_left)
            loss_bc_left = torch.mean((u_pred_bc_left - u_bc_left)**2)
            
            # Boundary condition loss (right)
            u_pred_bc_right = self.forward(x_bc_right, t_bc_right)
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
                print(f"Epoch {epoch+1}/{epochs} \
                      -- Total Loss: {loss.item():.6f} \
                      -- PDE Loss: {loss_pde.item():.6f} \
                      -- IC Loss: {loss_ic.item():.6f} \
                      -- BC Loss: {loss_bc.item():.6f}")
        
        return loss_history
    
class SimpleNN(nn.Module):
    """Simple Neural Network"""

    def __init__(self, layers, activation):
        super(SimpleNN, self).__init__()
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
    
    def train_nn(self, optimizer, data, nu, epochs, lambda_ic=10, lambda_bc=10, lambda_col=1, log_interval=100):
        """
        Train the SimpleNN model using supervised learning.
        
        Args:
            model: SimpleNN model
            optimizer: Optimizer
            data: Tuple of training data
            nu: Viscosity coefficient (unused, kept for compatibility)
            epochs: Number of training epochs
            lambda_ic: Weight for initial condition loss
            lambda_bc: Weight for boundary condition loss
            lambda_col: Weight for collocation points supervised loss
            log_interval: Interval for logging training progress
        """
        (x_init, t_init, u_init,
        x_bc_left, t_bc_left, u_bc_left,
        x_bc_right, t_bc_right, u_bc_right,
        x_col, t_col, u_col) = data
        
        loss_history = {'total': [], 'pde': [], 'ic': [], 'bc': []}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Supervised learning at collocation points (data-driven, not physics-based)
            u_pred_col = self.forward(x_col, t_col)
            loss_col = torch.mean((u_pred_col - u_col)**2)
            
            # Initial condition loss
            u_pred_init = self.forward(x_init, t_init)
            loss_ic = torch.mean((u_pred_init - u_init)**2)
            
            # Boundary condition loss (left)
            u_pred_bc_left = self.forward(x_bc_left, t_bc_left)
            loss_bc_left = torch.mean((u_pred_bc_left - u_bc_left)**2)
            
            # Boundary condition loss (right)
            u_pred_bc_right = self.forward(x_bc_right, t_bc_right)
            loss_bc_right = torch.mean((u_pred_bc_right - u_bc_right)**2)
            
            loss_bc = loss_bc_left + loss_bc_right
            
            # Total loss (SimpleNN uses supervised learning, NOT PDE residuals)
            loss = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_col * loss_col
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            loss_history['total'].append(loss.item())
            loss_history['pde'].append(loss_col.item())  # Store collocation loss as 'pde' for compatibility
            loss_history['ic'].append(loss_ic.item())
            loss_history['bc'].append(loss_bc.item())
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} \
                      -- Total Loss: {loss.item():.6f} \
                      -- Collocation Loss: {loss_col.item():.6f} \
                      -- IC Loss: {loss_ic.item():.6f} \
                      -- BC Loss: {loss_bc.item():.6f}")

        
        return loss_history