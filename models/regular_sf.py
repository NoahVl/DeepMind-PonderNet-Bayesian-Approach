import torch
from torch import nn


class RegularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        dims = [in_dim] + hidden_dims + [out_dim]
        layers = [nn.Linear(dims[0], dims[1])]
        for in_, out_ in zip(dims[1:], dims[2:]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_, out_))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, state=None):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        if state is None:
            # Create initial state
            state = x.new_zeros(batch_size, self.state_dim)

        y_hat, state = self.layers(torch.concat((x, state), dim=1)).tensor_split(
            indices=(self.out_dim,),
            dim=1,
        )

        return y_hat, state


class RegularRNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        state_dim: int,
        rnn_type: str = "rnn",
        activation: str = "relu",
    ):
        super().__init__()
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.rnn_type = rnn_type.lower()

        self.activation = {
            "relu": torch.relu,
            "tanh": torch.tanh,
        }[activation]

        rnn_cls = {
            "rnn": nn.RNNCell,
            "gru": nn.GRUCell,
        }[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=state_dim,
        )
        self.projection = nn.Linear(
            in_features=state_dim,
            out_features=out_dim,
        )

    def forward(self, x, state=None):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        state = self.rnn(x, state)
        state = self.activation(state)

        y_hat = self.projection(state)

        return y_hat, state
