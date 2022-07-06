# [UvA MSc AI DL2] Project: Can Neural Networks Estimate Difficulty?

This code supports our report of the same name.

## Authors

- Rick Akkerman
- Konrad Bereda
- Noah van der Vleuten
- Jeroen Wijnen 
- Stefan Wijnja

## Setup

Configure your virtual environment as desired using the conda environment or
`requirements.txt` as supplied in the `environments` directory.

The setup required to generate or gather data is done at runtime.

## Usage

- **TL;DR: Copy files from `train-presets` to the root directory of this
  repository to reproduce the experiments from our report.**

Read on to understand the configuration options.

Many projects use a command-line arguments to configure and run experiments.
Our project uses a `train.py` file in the root of the repository to configure
an experiment, and provides templates that reproduce the experiments from our
paper in the `train-presets/` directory.

Each training file does the following:

1. Configure a datamodule for a particular dataset. This is one of the
   datamodules in `datamodules.py`.
    - `TinyImageNet200Datamodule`: ImageNet, but scaled down to $64 \times 64$
      images of 200 classes.
    - `ParityDatamodule`: The parity problem as described in our report.
    - `FashionMNISTDataModule`: the [Fashion
      MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Results
      on this dataset are omitted in our report but fully supported in the
      codebase.
2. Configure the model that should be used. This can be `PonderNet`, which
   supports the PonderNet as proposed by Banino et al., and our Bayesian
   generalization, or `RegularNet`, which takes similar arguments but runs for
   a fixed number of 'pondering' steps. Arguments are described in full in the
   docstring for these modules, and example configuration can be found in
   `train-presets/`. The arguments are:

    - `step_function`: Which step function to use. One of [`mlp`, `rnn`,
      `bay_mlp`, `bay_rnn`]. The step functions that start with `bay_` use our
      bayesian generalization.
    - `step_function_args`: Dict of arguments to pass to the step function.
      This should contain configuration like:
        - `in_dim`: input dimensionality.
        - `out_dim`: output dimensionality.
        - `state_dim`: the dimensionality of the state that is passed from one
          step to another.
        - `rnn_type` (`rnn` and `bay_rnn` only): select what kind of RNN to
          use: `rnn` or `gru`.
    - `preds_reduction_method`: How the final prediction is assembled. One of:
        - `ponder`: Use the prediction where the Bernoulli draw parameterized
          by the halting parameter at that step lands on *halt*.
        - `bayesian`: Use a weighted average of the predictions at all steps,
          where the weights are determined by the probability of reaching a
          particular step and then stopping there.
        - `bayesian_sampling`: Same as above but with sampling. See report for
          details.
    - `lambda_prior` (PonderNet) or `beta_prior` (Bayesian PonderNet):
      parameterize the prior distribution.
    - `scale_reg`: the factor that the regularization term in the loss is
      scaled with.
    - `ponder_epsilon`: determines the threshold that determines the max
      probability of having stopped.
    - `learning_rate`: Learning rate for the model update.

    See the example configurations in `train-presets`, or read the docstring
    for more.

3. Configure the trainer object. This is not particular to our project, it's
   simply the Pytorch Lightning `Trainer`. However, you can use our
   regularization warmup callback that is located in `utils.py`. It takes a
   `start` and `slope` argument to configure the trajectory of the
   regularization warmup.

4. Call `trainer.fit`: start the configured experiment.
5. Call `trainer.test`: test the trained model on the test set.
