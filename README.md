# Higher Order QBM with GQSP

This repository contains experimental code for discriminative higher-order
Quantum Boltzmann Machines (QBMs) trained with Gibbs-state estimates from
Generalized Quantum Eigenvalue Transformation (GQEVT). The implementation is
intended to support the experiments described in the accompanying paper on
higher-order QBMs with GQSP/GQEVT-based Gibbs-state preparation.

The central training idea is:

1. Build an input-clamped Hamiltonian `H(x)`.
2. Build an input-and-label-clamped Hamiltonian `H(x, y)`.
3. Estimate expectation values under both Gibbs states.
4. Update the trainable Hamiltonian parameters from the difference between the
   clamped and unclamped expectations.

## Repository Structure

- `main.py` is the main entry point for a single experiment. It parses command
  line arguments, loads data, creates a QBM, trains it, and saves the resulting
  accuracy list and final weights.
- `data_loady.py` contains dataset loading and preprocessing. It currently
  supports two synthetic datasets (`circles`, `moons`) and a small UCI heart
  disease example.
- `QBM.py` defines the discriminative QBM model. It owns the training loop,
  prediction logic, negative log-likelihood tracking, and the contrastive
  update rule.
- `hamiltonian.py` defines `ModelHamiltonian`, which creates the trainable
  Pauli-term Hamiltonian and constructs the clamped Hamiltonians used during
  training.
- `GQEVT.py` implements the GQEVT/GQSP circuit machinery. It builds
  block-encodings, computes GQSP angles from polynomial coefficients, and
  provides different simulation levels.
- `helper.py` contains polynomial and angle helper routines used by `GQEVT.py`.
- `slurm_hyper.sh`, `slurm_hyperparam_single_seed.sh`, `slurm_single_seed.py` and
  `configure_hyperparams.py` are helper files for larger SLURM or hyperparameter
  runs.


## Installation

A conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate qbm-env
```

The equivalent Python dependencies are listed in `requirements.txt`.

## Running a Single Experiment

The default run uses the circles dataset and the Hamiltonian terms configured in
`main.py`:

```bash
python main.py
```

Useful command line options:

```bash
python main.py --dataset circles --epochs 20
python main.py --dataset moons --seed 478206
python main.py --connectivity all
python main.py --connectivity vh_only
python main.py --sim_level 0
python main.py --sim_level 2
python main.py --track_nll
```

The main defaults are intentionally collected in the argument parser in
`main.py`, so new experiments can usually be tested by changing command line
arguments rather than editing the model code.

## Model Configuration

The default Hamiltonian terms are:

```python
["Z", "X", "XX", "ZZ", "ZZZ"]
```

The current single-experiment setup uses:

- `n_hidden = 2`
- `n_output_qubits = 2`
- `n_visible = num_features + n_output_qubits`


`connectivity` controls which Pauli-word connections are included:

- `all`: include all valid hidden/visible connections allowed by
  `ModelHamiltonian`.
- `vh_only`: restrict multi-qubit terms to visible-visible nad visible-hidden style connections,
  avoiding hidden-hidden couplings.

## Simulation Levels

`GQEVT.py` supports three simulation levels:

- `sim_level = 0`: uses an exact matrix exponential/block-encoding path. This
  is useful as a small-system reference.
- `sim_level = 1`: builds an explicit matrix representation of the GQSP
  sequence and applies it as one unitary.
- `sim_level = 2`: constructs the GQSP circuit from rotations, controlled
  block-encodings, and reflections. Avoids nested optimization loop by defining a threshold when GQSP angles should be recalculated. Needs small learning rates to avoid reinitializing GQSP for every data sample


## Data Flow

`data_loady.py` returns a `DatasetSplit` object with:

- `train_x`, `train_y`
- `val_x`, `val_y`
- `test_x`, `test_y`

Synthetic data is scaled to `[-1, 1]`, shuffled with the requested seed, and
labels are one-hot encoded. The UCI loader performs basic cleaning,
one-hot-encoding of categorical features, conversion of binary features to
`{-1, +1}`, and min-max scaling of continuous features.

## Training Flow

The training loop in `QBM.train_model(...)` performs batched updates:

1. For each batch, build the clamped Hamiltonians through `ModelHamiltonian`.
2. Use `GQEVT` to estimate the relevant expectation values.
3. Match the expectation values back to the trainable parameter ordering.
4. Average the contrastive update over the batch.
5. Update the trainable weights and report test accuracy at the end of each
   epoch.

Negative log-likelihood tracking is optional from `main.py` because it requires
extra per-sample model evaluations:

```bash
python main.py --track_nll
```

## Relation to the Paper

The code follows the paper's description of a Higher-Order Quantum Boltzmann Machines (HOQBMs):
visible units encode inputs and labels, hidden units represent latent degrees of
freedom, and the Hamiltonian may include higher-order Pauli interactions such as
`ZZZ`.

The GQEVT component follows the paper's idea of approximating Gibbs-state
preparation by applying a polynomial transformation to a block-encoded
Hamiltonian without adding a nested optimization loop. In the notation of the paper, GQSP implements a transformation of
the normalized Hamiltonian, written as approximately `p(H / alpha)`, where
`alpha` is a block-encoding normalization. For a Gibbs operator, the polynomial
must be chosen so that this approximates the desired imaginary-time evolution,
for example `exp(-beta H)` or `exp(-beta H / 2)`, depending on the convention
used in the experiment.

## Outputs

By default, results are saved under `Results/`:

- `acc_list_seed<seed>.pkl`: test accuracy after each epoch.
- `weights_seed<seed>.pkl`: final trained Hamiltonian weights.

Use `--path` to choose another output directory.


