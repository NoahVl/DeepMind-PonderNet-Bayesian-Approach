import torch
from torch import nn


class PonderMLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: list[int], out_dim: int, state_dim: int
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.state_dim = state_dim

        dims = [in_dim + state_dim] + hidden_dims + [out_dim + state_dim + 1]
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

        y_hat_n, state, lambda_n = self.layers(
            torch.concat((x, state), dim=1)
        ).tensor_split(
            indices=(
                self.out_dim,
                self.out_dim + self.state_dim,
            ),
            dim=1,
        )

        lambda_n = lambda_n.squeeze().sigmoid()

        return y_hat_n, state, lambda_n, (torch.tensor(0), torch.tensor(0))


class PonderRNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        state_dim: int,
        rnn_type: str = "rnn",
        activation: str = "tanh",
    ):
        super().__init__()
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.rnn_type = rnn_type.lower()

        total_out_dim = out_dim + 1

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
            out_features=total_out_dim,
        )

    def forward(self, x, state=None):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        state = self.rnn(x, state)
        state = self.activation(state)

        y_hat_n, lambda_n = self.projection(state).tensor_split(
            indices=(self.out_dim,),
            dim=1,
        )

        lambda_n = lambda_n.squeeze().sigmoid()

        return y_hat_n, state, lambda_n, (torch.tensor(0), torch.tensor(0))
