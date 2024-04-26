import torch
import torch.nn.functional as F


# The abstract class for generating ODE-based data using PyTorch tensors
class AbstractODETarget:
    def __init__(self, dt=1e-3, t_step=0.25, dim=2):
        self.dim = dim
        self.dt = dt
        self.t_step = t_step
        self.n_step = int(t_step / dt)

    # Method to generate data with PyTorch tensors
    def generate_data(self, n_traj, traj_len):
        # Generating random initial conditions
        x0 = torch.rand(n_traj, self.dim) * (self.x_max - self.x_min) + self.x_min

        # Append the initial conditions and generate the trajectories
        data_x = torch.zeros(traj_len, n_traj, self.dim)  # Preallocate memory
        data_x[0] = x0  # Set initial conditions

        for t in range(1, traj_len):
            data_x[t] = self.euler(data_x[t - 1])  # Populate data

        # Stack the trajectories and transpose them to get the desired format
        data_x = data_x.transpose(0, 1)  # Transpose to [batch, time, features]

        return data_x

    # Right-hand side (RHS) function to be implemented by subclasses
    def rhs(self, x):
        raise NotImplementedError("RHS function must be implemented in subclass.")

    # Euler's method for solving the ODE
    def euler(self, x):
        # Iterating Euler's method n_step times
        for _ in range(self.n_step):
            x += self.dt * self.rhs(x)
        return x


# Class representing a Duffing Oscillator with PyTorch
class DuffingOscillator(AbstractODETarget):
    def __init__(self, dt=1e-3, t_step=0.25, dim=2, delta=0.5, alpha=1.0, beta=-1.0):
        super().__init__(dt, t_step, dim)
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.x_min = -2
        self.x_max = 2

    # Override the rhs method with the Duffing oscillator's equations
    def rhs(self, x):
        x1 = x[:, 1].unsqueeze(1)  # Reshape for proper concatenation
        x2 = x[:, 0].unsqueeze(1)

        # Define the right-hand side equations
        f1 = x2
        f2 = -self.delta * x2 - x1 * (self.beta + self.alpha * x1**2)

        # Concatenate the results to get the updated values for both states
        return torch.cat([f1, f2], dim=-1)
