import torch
import math
from torch import nn
from lightning import LightningModule

class ConstantMatrixMultiplier(LightningModule):
    def __init__(self, n_psi):
        super(ConstantMatrixMultiplier, self).__init__()
        """
        Initialize K as a n_spi x n_psi trainable parameter
        """        

        self.K = nn.Parameter(torch.empty(n_psi, n_psi), requires_grad=False)
        nn.init.kaiming_normal_(self.K, a=math.sqrt(5))

        self.save_hyperparameters()
    def forward(self, inputs):
        """ Perform matrix multiplication

        Args:
            inputs (_type_): (batch_size, n_psi)

        Returns:
            _type_: (batch_size, n_psi)
        """        
       
        return torch.matmul(inputs, self.K)