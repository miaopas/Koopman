from mamba_ssm import Mamba as MambaBlock
from torch import nn


class Mamba(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_layers=1, **kwarg):
        super().__init__()
        layers = [MambaBlock(**kwarg) for _ in range(num_layers)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.network(x)

        return x
