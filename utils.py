import torch
from pytorch_lightning.callbacks import Callback


def calculate_beta_std(alphas: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """
    Calculate the standard deviation of the beta distribution.

    :return: Standard deviation of the beta distribution.
    """
    variance = (alphas * betas) / ((alphas + betas) ** 2 * (alphas + betas + 1))
    return torch.sqrt(variance)


def mode_agreement_metric(samples_mode, samples):
    """
    Calculate the mode agreement metric.

    :param samples_mode: The mode of each sample. (batch_size)
    :param samples: The samples. (batch_size, samples)
    :return: The normalized agreement between the samples and the mode. (float)
    """
    return (
        (samples == samples_mode.unsqueeze(1)).float().sum(1) / samples.size(1)
    ).mean()


class RegularizationWarmup(Callback):
    """Warm up a regularization factor."""

    def __init__(self, start: float, slope: float, model_attr: str):
        """
        Args:
            start: value for the warmup factor at the start of training. If
                this value is > 1 the slope should be in [0,1] and warmup stops
                when the factor dips below 1. If this value is in [0, 1] the
                slope should be > 1 and warmup stops when the factor exceeds 1.
            slope: increment the warmup factor by this value at the end of
                every epoch. Can be between 0 and 1 or greater than 1.
            model_attr: name of the property on the pl_module that is being
                trained that the warmup factor should be saved to at the start
                of every epoch.
        """
        self.start = start
        self.slope = slope
        self.attr_name = model_attr

        if start < 0 or slope < 0:
            raise ValueError("Regularization warmup settings shouldn't be negative.")
        if (start > 1 and slope > 1) or (start < 1 and slope < 1):
            raise ValueError("Regularization warmup never ends with these settings.")

    def on_fit_start(self, trainer, pl_module):
        setattr(pl_module, self.attr_name, self.start)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.fit_loop.epoch_progress.total.completed != 0:
            current = getattr(pl_module, self.attr_name)
            new = current * self.slope
            if self.start >= 1 and new <= 1:
                new = 1
            elif self.start < 1 and new >= 1:
                new = 1
            setattr(pl_module, self.attr_name, new)

            for logger in trainer.loggers:
                logger.log_metrics(
                    {self.attr_name: new},
                    step=trainer.fit_loop.epoch_loop._batches_that_stepped,
                )
        else:
            for logger in trainer.loggers:
                logger.log_metrics(
                    {self.attr_name: self.start},
                    step=trainer.fit_loop.epoch_loop._batches_that_stepped,
                )
