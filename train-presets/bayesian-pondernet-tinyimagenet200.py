# stdlib
import os

# third party
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# first party
import datamodules
import models


def main():
    seed = 420
    pl.seed_everything(seed)
    datamodule = datamodules.TinyImageNet200DataModule(
        num_workers=os.cpu_count(), batch_size=128
    )
    model = models.PonderNet(
        encoder="efficientnet",
        encoder_args=dict(
            variant=0,
        ),
        step_function="bay_rnn",
        step_function_args=dict(
            in_dim=1280,
            out_dim=datamodule.num_classes,
            state_dim=500,
            rnn_type="gru",
        ),
        beta_prior=(3, 3),
        max_ponder_steps=10,
        preds_reduction_method="bayesian_sampling",
        task="classification",
        learning_rate=3e-4,
        scale_reg=0.01,
        ponder_epsilon=0.05,
        # Extra args just to log them
        dataset=type(datamodule).__name__,
        seed=seed,
        regularization_warmup=False,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                monitor="acc/val",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        deterministic=True,
        devices="auto",
        logger=[
            TensorBoardLogger(
                "logs",
                default_hp_metric=True,
            ),
            # Uncomment to enable logging to Weights & Biases
            # WandbLogger(
            #     name="Bayesian PonderNet",
            #     entity="mscai-dl2",
            #     project="mscai-dl2",
            #     log_model=True,
            # ),
        ],
        max_epochs=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
