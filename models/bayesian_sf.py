import torch
import torch.distributions.beta as dist_beta
import torch.nn.functional as F
from torch import nn


class PonderBayesianMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        state_dim: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.state_dim = state_dim

        total_out_dim = (
            out_dim + state_dim + 2
        )  # add two extra items for dimension for alpha and beta
        if hidden_dims:
            layers: list[nn.Module] = [nn.Linear(in_dim + state_dim, hidden_dims[0])]
            for in_, out in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers += [nn.ReLU(), nn.Linear(in_, out)]
            layers += [nn.ReLU(), nn.Linear(hidden_dims[-1], total_out_dim)]
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Sequential(nn.Linear(in_dim + state_dim, total_out_dim))

    def forward(self, x, state=None):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        if state is None:
            # Create initial state
            state = x.new_zeros(batch_size, self.state_dim)

        y_hat_n, state, lambda_params = self.layers(
            torch.concat((x, state), dim=1)
        ).tensor_split(
            indices=(
                self.out_dim,
                self.out_dim + self.state_dim,
            ),
            dim=1,
        )

        # 2) Sample lambda_n from beta-distribution
        lambda_params = F.relu(lambda_params) + 1e-7
        alpha, beta = lambda_params[:, 0], lambda_params[:, 1]
        distribution = dist_beta.Beta(alpha, beta)
        lambda_n = distribution.rsample()

        return y_hat_n, state, lambda_n, (alpha, beta)


class PonderBayesianRNN(nn.Module):
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

        total_out_dim = out_dim + 2

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

        y_hat_n, lambda_params = self.projection(state).tensor_split(
            indices=(self.out_dim,),
            dim=1,
        )

        # 2) Sample lambda_n from beta-distribution
        lambda_params = F.relu(lambda_params) + 1e-7
        alpha, beta = lambda_params[:, 0], lambda_params[:, 1]
        distribution = dist_beta.Beta(alpha, beta)
        lambda_n = distribution.rsample()

        return y_hat_n, state, lambda_n, (alpha, beta)
