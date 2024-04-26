import torch
import torch.nn.utils.parametrize as P
from torch import nn


class Linear(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 128,
        activation: str = "linear",
        bias: bool = True,
    ):
        super().__init__()

        self.U = nn.Linear(input_size, output_size, bias=bias)
        self.input_size = input_size

        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh
        elif activation == "softmax":
            self.activation = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_size = x.size()[-1]
        assert input_size == self.input_size

        x = self.U(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    test_model = Linear(2, 3)

    inputs = torch.randn(1, 10, 2)
    outputs = test_model(inputs)
    assert outputs.shape == (1, 10, 3)
    print("Test passed.")
