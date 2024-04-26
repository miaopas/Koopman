from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialModel(nn.Module):
    """A wrapper for sequential type model with form layer1 -> layer2 -> ...

    -> layern
    """

    def __init__(self, layers: Sequence[nn.Module] = None):  # currently useless
        super().__init__()
        if layers is None:
            raise ValueError("The 'layers' argument cannot be None.")
        self.layers = nn.ModuleList(layers)
        # self.intermediates = []

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for layer in self.layers:
            x = layer(x)
            # self.intermediates.append(x)

            # Here assumes if there are extra outputs, it is a tuple
            if isinstance(x, tuple):
                x = x[0]

        return x
