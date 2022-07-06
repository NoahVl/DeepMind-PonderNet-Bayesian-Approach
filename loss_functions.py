from typing import Callable

import torch
import torch.distributions.beta as dist_beta
from torch import Tensor, lgamma, nn


class PonderLoss(nn.Module):
    def __init__(
        self,
        task_loss_fn: Callable,
        scale_reg: float,
        lambda_prior: float,
        max_ponder_steps: int,
    ):
        """
        Args:
            scale_reg: Weight for the regularization loss term.
            lambda_reg: Parameterizes the (Bernoulli) prior.
            task_loss_fn: Loss function for the actual task (e.g. MSE or CE).
            max_ponder_steps
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.scale_reg = scale_reg
        self.lambda_prior = lambda_prior
        self.KL = nn.KLDivLoss(reduction="batchmean")

        prior = lambda_prior * (1 - lambda_prior) ** torch.arange(max_ponder_steps)
        prior = prior / prior.sum()
        self.register_buffer("log_prior", prior.log())

    def forward(
        self,
        preds: Tensor,
        p: Tensor,
        halted_at: Tensor,
        targets: Tensor,
        regularization_warmup_factor: float,
        **kwargs
    ):
        """
        Args:
            `preds`: Predictions of shape (ponder_steps, batch_size, logits)
            `p`: Cumulative probability of reaching and then stopping at each
                step of shape (step, batch)
            `halted_at`: Indices of steps where each sample actually stopped of
                shape (batch)
            `targets`: Targets of shape (batch_size)
            `regularization_warmup_factor`: Factor used to warm up
                regularization loss term.
        """
        n_steps, batch_size, _ = preds.shape

        # Reconstruction term
        task_losses = self.task_loss_fn(
            preds.view(
                -1, preds.size(-1)
            ),  # View pred steps as individual classifications.
            targets.repeat(n_steps),  # Repeat targets as needed to match.
            reduction="none",
        ).view(n_steps, batch_size)
        l_rec = (task_losses * p).sum(0).mean()

        # Regularization term
        p_t = p.transpose(1, 0)
        l_reg = self.KL(self.log_prior[:n_steps].expand_as(p_t), p_t)

        return l_rec, regularization_warmup_factor * self.scale_reg * l_reg


class PonderBayesianLoss(nn.Module):
    def __init__(
        self,
        task_loss_fn: Callable,
        beta_prior: tuple[float, float],
        max_ponder_steps: int,
        scale_reg: float,
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.beta_prior = beta_prior
        self.KL = nn.KLDivLoss(reduction="none")
        self.scale_reg = scale_reg

        self.prior = dist_beta.Beta(beta_prior[0], beta_prior[1])

    def forward(
        self,
        preds: Tensor,
        p: Tensor,
        halted_at: Tensor,
        targets: Tensor,
        regularization_warmup_factor: float,
        **kwargs
    ):
        """
        Args:
            `preds`: Predictions of shape (ponder_steps, batch_size, logits)
            `p`: Cumulative probability of reaching and then stopping at each
                step of shape (step, batch)
            `halted_at`: Indices of steps where each sample actually stopped of
                shape (batch)
            `targets`: Targets of shape (batch_size)
            'lambdas': Lambdas of shape (step, batch_size)
            `regularization_warmup_factor`: Factor used to warm up
                regularization loss term.
        """
        assert "lambdas" in kwargs, "Must provide lambdas!"

        lambdas = kwargs["lambdas"]
        alphas, betas = kwargs["beta_params"]

        n_steps, batch_size, _ = preds.shape

        # Reconstruction term
        task_losses = self.task_loss_fn(
            preds.view(
                -1, preds.size(-1)
            ),  # View pred steps as individual classifications.
            targets[
                torch.arange(targets.size(0)).repeat(n_steps)
            ],  # Repeat targets as needed to match.
            reduction="none",
        ).view(n_steps, batch_size)
        l_rec = torch.einsum("ij,ij->j", p, task_losses).mean()

        # Regularization term
        # TODO : Make hyperparameter to decide if you want to use approximation
        # l_reg_alt = (
        #     self.KL(
        #         self.prior.rsample(sample_shape=(batch_size, n_steps))
        #         .to(lambdas.device)
        #         .log(),
        #         lambdas.transpose(1, 0),  # type: ignore
        #     )
        #     .sum(1)
        #     .mean()
        # )  # Sum over the number of steps, then mean over the batch.

        def lbeta(x, y):
            # As derivable from:
            # https://en.wikipedia.org/wiki/Beta_function#:~:text=.%5B1%5D-,A%20key%20property,-of%20the%20beta
            return lgamma(x) + lgamma(y) - lgamma(x + y)

        a_prime, b_prime = torch.Tensor(self.beta_prior).to(lambdas.device)
        a, b = alphas, betas
        # Analytically computing KL-divergence, according to formula in
        # https://en.wikipedia.org/wiki/Beta_distribution#:~:text=The%20relative%20entropy%2C%20or%20Kullback%E2%80%93Leibler%20divergence%20DKL(X1%20%7C%7C%20X2)
        l_reg = (
            (
                lbeta(a_prime, b_prime)
                - lbeta(a, b)
                + (a - a_prime) * torch.digamma(a)
                + (b - b_prime) * torch.digamma(b)
                + (a_prime - a + b_prime - b) * torch.digamma(a + b)
            )
            .sum(0)
            .mean()
        )

        return l_rec, regularization_warmup_factor * self.scale_reg * l_reg
