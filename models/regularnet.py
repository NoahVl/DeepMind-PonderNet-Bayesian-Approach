from typing import Literal, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import optim

# First party
from .encoders import EfficientNetEncoder
from .pondernet import PonderNet
from .regular_sf import RegularMLP, RegularRNN


class RegularNet(LightningModule):
    """Network to configure baselines of the same size as a PonderNet, but
    with a more efficient compute graph because the number of steps is
    a hyperparameter and thus deterministic."""

    def __init__(
        self,
        *,
        step_function: Literal["mlp", "rnn"],
        step_function_args: dict,
        task: Literal["classification"],
        encoder: Optional[Literal["efficientnet"]] = None,
        encoder_args: Optional[dict] = None,
        learning_rate: float = 3e-4,
        fixed_ponder_steps: int = 1,
        **kwargs,  # Just to log them.
    ):
        """
        Args:
            step_function: Which step function to use.
            step_function_args: Arguments to pass to the step function module.
            task: Which task to perform (originally intended to support
                regression).
            encoder: Name of the encoder to use.
            encoder_args: Arguments to pass to the encoder module.
            learning_rate: Learning rate to use during training.
            fixed_ponder_steps: Loop over the step function for this number of
                steps. Always used in this RegularNet.
        """
        super().__init__()
        self.save_hyperparameters()

        assert (
            fixed_ponder_steps > 0
        ), "Fixed number of ponder steps has to be > 0 in RegularNet"

        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        # Encoder
        if encoder:
            encoder_class = {
                "efficientnet": EfficientNetEncoder,
            }.get(encoder)
            if not encoder_class:
                raise ValueError(f"Unknown encoder: '{encoder}'")
            self.encoder = encoder_class(**encoder_args)
        else:
            self.encoder = lambda x: x  # type: ignore

        # Step function
        sf_class = {
            "mlp": RegularMLP,
            "rnn": RegularRNN,
        }.get(step_function)
        if not sf_class:
            raise ValueError(f"Unknown step function: '{step_function}'")
        self.step_function = sf_class(**step_function_args)

        # Loss
        lf_class = {
            "classification": F.cross_entropy,
        }.get(task)
        if not lf_class:
            raise NotImplementedError(f"Unknown task: '{task}'")
        self.loss_function = lf_class

        # Optimizer
        self.optimizer_class = optim.Adam

        # Default so that this model can be used without regularization warmup.
        self.regularization_warmup_factor = 1

    def configure_optimizers(self):
        return PonderNet.configure_optimizers(self)

    def forward(self, x):
        state = None  # State that transfers across steps
        x = self.encoder(x)

        for _ in range(self.hparams.fixed_ponder_steps):
            y_hat, state = self.step_function(x, state)

        return y_hat  # (batch, logit)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.loss_function(preds, targets)

        self.train_acc(preds, targets)
        self.log("acc/train", self.train_acc, on_step=False, on_epoch=True)
        self.log("loss/train", loss)

        self.log(
            "halted_at/mean/train",
            float(self.hparams.fixed_ponder_steps),
            on_step=True,
            on_epoch=True,
        )
        self.log("halted_at/std/train", 0.0, on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/train",
            float(self.hparams.fixed_ponder_steps),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.loss_function(preds, targets)

        self.val_acc(preds, targets)
        self.log("loss/val", loss)
        self.log("acc/val", self.val_acc, on_step=False, on_epoch=True)

        self.log(
            "halted_at/mean/val",
            float(self.hparams.fixed_ponder_steps),
            on_step=True,
            on_epoch=True,
        )
        self.log("halted_at/std/val", 0.0, on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/val",
            float(self.hparams.fixed_ponder_steps),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.loss_function(preds, targets)

        self.test_acc(preds, targets)
        self.log("loss/test", loss)
        self.log("acc/test", self.test_acc, on_step=False, on_epoch=True)

        self.log(
            "halted_at/mean/test",
            float(self.hparams.fixed_ponder_steps),
            on_step=True,
            on_epoch=True,
        )
        self.log("halted_at/std/test", 0.0, on_step=True, on_epoch=True)
        self.log(
            "halted_at/median/test",
            float(self.hparams.fixed_ponder_steps),
            on_step=True,
            on_epoch=True,
        )
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_epoch_start_time = torch.cuda.Event(enable_timing=True)
        self.test_epoch_start_time.record()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        out = super().on_test_epoch_end()
        test_epoch_end_time = torch.cuda.Event(enable_timing=True)
        test_epoch_end_time.record()
        torch.cuda.synchronize()
        curr_time = self.test_epoch_start_time.elapsed_time(test_epoch_end_time)
        self.log("efficiency/test_time", curr_time)
        return out
