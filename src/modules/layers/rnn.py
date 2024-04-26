import torch
from torch import nn


class LinearRNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hid_dim,
    ):
        super().__init__()

        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.hidden_ff = nn.Linear(hid_dim, hid_dim)
        self.output_ff = nn.Linear(hid_dim, output_dim)

        # Diag initialize
        self.hidden_ff.weight.data = torch.diag(torch.rand(hid_dim))

        # register h0
        self.register_buffer("h0", torch.zeros(1, 1, hid_dim))

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # src = [batch size, input len, input dim]
        length = x.shape[1]

        hidden = []
        hidden.append(self.h0)

        x = self.input_ff(x)

        for i in range(length):
            h_next = x[:, i : i + 1, :] + self.hidden_ff(hidden[i])
            hidden.append(h_next)

        hidden = torch.cat(hidden[1:], dim=1)
        out = self.output_ff(hidden)
        return out
