import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule


class DicNN(LightningModule):
    """Trainable dictionaries."""

    def __init__(self, inputs_dim=1, layer_sizes=[64, 64], n_psi_train=22, activation_func="tanh"):
        super(DicNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.layer_sizes = layer_sizes
        self.n_psi_train = n_psi_train
        self.activation_func = activation_func

        # Creating the input layer
        self.input_layer = nn.Linear(self.inputs_dim, layer_sizes[0], bias=False)

        # Creating hidden layers
        self.hidden_layers = nn.ModuleList()
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.hidden_layers.append(nn.Linear(in_features, out_features))

        # Creating the output layer
        self.output_layer = nn.Linear(layer_sizes[-1], n_psi_train)
        self.save_hyperparameters()
    def forward(self, inputs):
        # Check layer dimension
        if inputs.shape[-1] != self.inputs_dim:
            print(f"Error: Expected input dimension {self.inputs_dim}, but got {inputs.shape[-1]}")
            return None  # Optionally, you could raise an exception here

        # Apply the input layer
        psi_x_train = self.input_layer(inputs)

        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            if self.activation_func == "tanh":
                psi_x_train = psi_x_train + F.tanh(layer(psi_x_train))
            elif self.activation_func == "relu":
                psi_x_train = psi_x_train + F.relu(layer(psi_x_train))
            else:
                raise ValueError("Unsupported activation function")

        # Apply the output layer
        outputs = self.output_layer(psi_x_train)
        return outputs


class PsiNN(nn.Module):
    def __init__(
        self,
        inputs_dim=1,
        dic_trainable=DicNN,
        layer_sizes=[64, 64],
        n_psi_train=22,
        activation_func="tanh",
        add_constant=True,
    ):
        super(PsiNN, self).__init__()
        self.n_psi_train = n_psi_train
        self.add_constant = add_constant
        # Create an instance of the dic_trainable with given parameters
        self.dicNN = (
            dic_trainable(inputs_dim, layer_sizes, n_psi_train, activation_func)
            if n_psi_train != 0
            else None
        )

    def generate_B(self, inputs, add_constant=True):
        target_dim = inputs.shape[-1]  # Get the last dimension of the input tensor

        psi_dim = self.n_psi_train + target_dim + add_constant

        B = torch.zeros((psi_dim, target_dim), dtype=inputs.dtype)

        if add_constant:
            B[1:target_dim + 1, :target_dim].fill_diagonal_(1)
        else:
            B.fill_diagonal_(1)

        return B

    def forward(self, inputs):
        outputs = []

        # Add a constant column of ones
        if self.add_constant:
            constant = torch.ones_like(inputs)[..., [0]]
            outputs.append(constant)

        # Add the original inputs
        outputs.append(inputs)

        # Add the output from dicNN if applicable
        if self.n_psi_train != 0:
            psi_x_train = self.dicNN(inputs)
            outputs.append(psi_x_train)

        # Concatenate along the feature dimension
        outputs = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]

        return outputs