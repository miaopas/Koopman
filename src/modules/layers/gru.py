from torch import nn
from torch.nn import GRU as GRUBlock


class GRU(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_layers=1, **kwarg):
        super().__init__()
        self.layers = nn.ModuleList([GRUBlock(**kwarg) for _ in range(num_layers)])

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for layer in self.layers:
            x = layer(x)[0]

        return x


class GRU_(nn.Module):
    """_summary_
    Experimental, that each layer in between has a linear output
    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_layers=1, **kwarg):
        super().__init__()
        layers = []

        for _ in range(num_layers):
            layers.append(nn.Linear(kwarg["input_size"], kwarg["hidden_size"]))
            layers.append(GRUBlock(**kwarg))
            layers.append(nn.Linear(kwarg["hidden_size"], kwarg["input_size"]))

        self.layers = nn.ModuleList(layers)
        

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for i in range(0,len(self.layers), 3):
            x = self.layers[i](x)
            x = self.layers[i+1](x)[0]
            x = self.layers[i+2](x)

        return x
