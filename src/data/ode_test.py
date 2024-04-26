from scipy import signal
from scipy.fftpack import diff as psdiff
from scipy.integrate import solve_ivp
import numpy as np

class AbstractODETarget:
    def __init__(self, dt=1e-3, t_step=0.25, dim=2):
        self.dim = dim
        self.dt = dt
        self.t_step = t_step
        self.n_step = int(t_step / dt)

    def generate_init_data(self, n_traj, traj_len, seed=None):
        data_x = []
        if seed is not None:
            np.random.seed(seed)

        x0 = np.random.uniform(size=(n_traj, self.dim), low=self.x_min, high=self.x_max)

        data_x.append(x0)
        for t in range(traj_len - 1):
            data_x.append(self.euler(data_x[t]))

        data_x = np.asarray(data_x)

        data_x = np.transpose(data_x, [1, 0, 2]).reshape(n_traj * traj_len, self.dim)
        return np.asarray(data_x)

    def generate_next_data(self, data_x):
        data_y = self.euler(data_x)
        return data_y

    def generate_data(self, n_traj, traj_len, seed=None):
        data_x = []
        if seed is not None:
            np.random.seed(seed)

        x0 = np.random.uniform(size=(n_traj, self.dim), low=self.x_min, high=self.x_max)

        data_x.append(x0)
        for t in range(traj_len - 1):
            data_x.append(self.euler(data_x[t]))

        data_x = np.asarray(data_x)
        data_x = np.transpose(data_x, [1, 0, 2])
        return np.asarray(data_x)

    def rhs(self):
        """RHS Function :return: The rhs of one specific ODE."""
        return NotImplementedError

    def euler(self, x):
        """ODE Solver.

        :param x: variable
        :type x: vector (float)
        :return: ODE Solution at t_step after iterating the Euler method n_step times
        :rtype: vector with the same shape as the variable x (float)
        """
        for _ in range(self.n_step):
            x = x + self.dt * self.rhs(x)
        return x


class DuffingOscillator(AbstractODETarget):
    """Duffing equation based on the notation in.

    (https://en.wikipedia.org/wiki/Duffing_equation)
    """

    def __init__(self, dt=1e-3, t_step=0.25, dim=2, delta=0.5, alpha=1.0, beta=-1.0):
        super().__init__(dt, t_step, dim)
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.x_min = -2
        self.x_max = 2

    def rhs(self, x):
        x1 = x[:, 0].reshape(x.shape[0], 1)
        x2 = x[:, 1].reshape(x.shape[0], 1)
        f1 = x2
        f2 = -self.delta * x2 - x1 * (self.beta + self.alpha * x1**2)
        return np.concatenate([f1, f2], axis=-1)