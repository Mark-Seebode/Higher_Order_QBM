import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from QBM import QBM
    from data_loady import DatasetSplit


DEFAULT_TERMS = ["Z", "X", "XX", "ZZ", "ZZZ"]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    path: str
    run_name: str
    chebyshev_degree: int
    beta: float
    batch_size: int
    learning_rate: float
    gv_1: float
    gv_2: float
    connectivity: str
    sim_level: int
    h_norm_trigger: float
    h_norm_target: float
    track_nll: bool
    gqevt_refresh_mode: str
    dataset: str
    epochs: int


def build_qbm(config: ExperimentConfig, num_features: int) -> "QBM":
    from QBM import QBM

    n_hidden = 2
    n_output_qubits = 2
    n_visible = num_features + n_output_qubits

    print("terms:", DEFAULT_TERMS)
    print("Creating QBM...")
    return QBM(
        n_hidden=n_hidden,
        n_visible=n_visible,
        n_output=n_output_qubits,
        terms=DEFAULT_TERMS,
        connectivity=config.connectivity,
        n=config.chebyshev_degree,
        beta=config.beta,
        seed=config.seed,
        allow_visible_x=False,
        gamma_visible=[config.gv_1],
        sim_level=config.sim_level,
        h_norm_trigger=config.h_norm_trigger,
        h_norm_target=config.h_norm_target,
        track_nll=True,#config.track_nll,
        gqevt_refresh_mode=config.gqevt_refresh_mode,
    )


def train_qbm(config: ExperimentConfig, qbm: "QBM", data: "DatasetSplit"):
    print("Training QBM...")
    return qbm.train_model(
        data.train_x,
        data.train_y,
        data.test_x,
        data.test_y,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        plot=False,
    )


def save_results(config: ExperimentConfig, qbm: "QBM") -> None:
    output_dir = Path(config.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"acc_list_seed{config.seed}.pkl", "wb") as f:
        pickle.dump(qbm.accuracy_list, f)

    with open(output_dir / f"weights_seed{config.seed}.pkl", "wb") as f:
        pickle.dump(qbm.hamiltonian.get_weights(), f)


def run_experiment(config: ExperimentConfig) -> "QBM":
    from data_loady import load_dataset_split

    print(f"Loading {config.dataset} data for seed {config.seed}...")
    data = load_dataset_split(seed=config.seed, dataset=config.dataset)
    print(
        "Data shapes:",
        f"train={data.train_x.shape}",
        f"val={data.val_x.shape}",
        f"test={data.test_x.shape}",
    )

    qbm = build_qbm(config, num_features=data.num_features)
    train_qbm(config, qbm, data)

    print("QBM trained")
    print(f"Results for seed {config.seed}: \nACC={qbm.accuracy_list}")

    save_results(config, qbm)
    return qbm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one HOQBM experiment.")
    parser.add_argument("--seed", type=int, default=478206, help="Random seed for reproducibility")
    parser.add_argument("--path", type=str, default="Results/", help="Directory for result files")
    parser.add_argument(
        "--run_name",
        type=str,
        default="e20_b1_bs32_lr0.01",
        help="Name for this run. Kept for compatibility with older scripts.",
    )
    parser.add_argument(
        "--chebyshev_degree",
        type=int,
        default=100,
        help="Degree of the Chebyshev polynomial approximation",
    )
    parser.add_argument("--beta", type=float, default=5.939580503628701, help="Inverse temperature")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.007099297428361392, help="Learning rate")
    parser.add_argument("--gv_1", type=float, default=1.0, help="First visible gamma value")
    parser.add_argument(
        "--gv_2",
        type=float,
        default=1.0,
        help="Second visible gamma value. Parsed for compatibility; current single-seed model uses gv_1.",
    )
    parser.add_argument(
        "--connectivity",
        choices=["all", "vh_only"],
        default="all",
        help="Hamiltonian connectivity pattern",
    )
    parser.add_argument(
        "--sim_level",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="GQET simulation level",
    )
    parser.add_argument(
        "--h_norm_trigger",
        type=float,
        default=3.0,
        help="Rescale the QBM weights after a batch update when their L2 norm exceeds this value",
    )
    parser.add_argument(
        "--h_norm_target",
        type=float,
        default=1.0,
        help="When rescaling is triggered, reduce the QBM weight norm to this value",
    )
    parser.add_argument(
        "--track_nll",
        action="store_true",
        help="Track per-batch NLL during training. This is expensive for sim_level 1 or 2.",
    )
    parser.add_argument(
        "--gqevt_refresh_mode",
        choices=["threshold", "on_error"],
        default="threshold",
        help="Use threshold rescaling or rebuild GQEVT only after an evaluation error",
    )
    parser.add_argument(
        "--dataset",
        choices=["circles", "moons", "uci"],
        default="circles",
        help="Dataset to train on",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        seed=args.seed,
        path=args.path,
        run_name=args.run_name,
        chebyshev_degree=args.chebyshev_degree,
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gv_1=args.gv_1,
        gv_2=args.gv_2,
        connectivity=args.connectivity,
        sim_level=args.sim_level,
        h_norm_trigger=args.h_norm_trigger,
        h_norm_target=args.h_norm_target,
        track_nll=args.track_nll,
        gqevt_refresh_mode=args.gqevt_refresh_mode,
        dataset=args.dataset,
        epochs=args.epochs,
    )


def main(argv=None) -> "QBM":
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    return run_experiment(config)


if __name__ == "__main__":
    main()
