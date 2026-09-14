from sklearn.datasets import make_circles, make_moons
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import sklearn
import argparse
import os
import json
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd

from QBM import QBM


def make_data(seed: int, use_moons: bool = False):
    if use_moons:
        X, y = make_moons(n_samples=1000, noise=0.15, random_state=seed)
    else:
        X, y = make_circles(n_samples=1000, factor=0.1, noise=0.15, random_state=seed)

    X = X.astype(np.float32)
    y = y.astype(int).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X).astype(np.float32)
    X, y = sklearn.utils.shuffle(X, y, random_state=seed)

    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    y_oh = encoder.fit_transform(y)

    train_x, temp_x, train_y, temp_y = train_test_split(
        X, y_oh, test_size=0.3, random_state=seed, stratify=y_oh
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=seed, stratify=temp_y
    )

    return train_x, val_x, test_x, train_y, val_y, test_y

def make_uci_data(seed: int):
    heart_disease = fetch_ucirepo(id=45)
    X_df = heart_disease.data.features.copy()
    y_df = heart_disease.data.targets.copy()

    df = X_df.copy()
    df["target"] = y_df

    df.dropna(inplace=True)

    if isinstance(df["target"], pd.DataFrame):
        df["target"] = df["target"].iloc[:, 0]

    # Turn target into binary: 0 (no disease) vs 1 (disease)
    df["target"] = df["target"].apply(lambda x: 1 if int(x) > 0 else 0).astype(int)


    categorical_features = ["cp", "restecg", "slope", "ca", "thal"]

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    df["sex"] = df["sex"].apply(lambda v: 1 if int(v) == 1 else 0).astype(int)

    numerical_features = ["age", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak"]

    scaler_standard = StandardScaler()
    df[numerical_features] = scaler_standard.fit_transform(df[numerical_features])

    scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
    df[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])

    print("NaNs per column:\n", df.isnull().sum())
    print("Final input feature count:", df.drop("target", axis=1).shape[1])


    feature_cols = df.drop("target", axis=1).columns
    X = df.drop("target", axis=1).values.astype(np.float32)

    y = df["target"].values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    y = encoder.fit_transform(y)

    train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=seed, stratify=temp_y)

    return train_x, val_x, test_x, train_y, val_y, test_y


def build_qbm(seed: int, chebyshev_degree: int, beta: float, gv_1: float, gv_2: float, num_features: int = 2):
    terms = ["Z", "ZZ"]

    num_features = num_features
    n_hidden = 5
    n_output_qubits = 2
    n_visible = num_features + n_output_qubits

    print(f"Creating QBM for seed {seed}...")
    qbm = QBM(
        n_hidden=n_hidden,
        n_visible=n_visible,
        n_output=n_output_qubits,
        terms=terms,
        connectivity="all",  # "vh_only" or "all"
        n=chebyshev_degree,
        beta=beta,
        seed=seed,
        allow_visible_x=False,
        gamma_visible=[gv_1, gv_2],
    )
    return qbm


def to_cat(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(int)
    if y.shape[1] == 1:
        return y.reshape(-1).astype(int)
    return np.argmax(y, axis=1).astype(int)


def evaluate_test_accuracy(qbm, test_x, test_y):
    preds = []
    for x in test_x:
        _, pred = qbm.predict(x)
        preds.append(pred)
    return accuracy_score(to_cat(test_y), preds)


def train_one_seed(
    seed: int,
    chebyshev_degree: int,
    beta: float,
    batch_size: int,
    lr: float,
    gv_1: float,
    gv_2: float,
    best_epoch: int,
    use_moons: bool = False,
):
    train_x, val_x, test_x, train_y, val_y, test_y = make_data(seed, use_moons=use_moons)
    qbm = build_qbm(seed, chebyshev_degree, beta, gv_1, gv_2, num_features=train_x.shape[1])

    print(f"Training seed {seed} for {best_epoch} epochs...")
    # Capture the losses list here
    _, nll = qbm.train_model(
        train_x, train_y,
        test_x, test_y,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=best_epoch,
        plot=False,
    )

    acc_list = [float(x) for x in qbm.accuracy_list]
    test_acc = evaluate_test_accuracy(qbm, test_x, test_y)

    data = {
        "train_x": train_x,
        "val_x": val_x,
        "test_x": test_x,
        "train_y": train_y,
        "val_y": val_y,
        "test_y": test_y,
    }

    return qbm, test_acc, acc_list, nll, data

def plot_mean_decision_boundary(models, all_data, best_epoch, out_path, grid_points=220, pad=0.2):
    all_X = []
    all_y = []

    for data in all_data:
        X_vis = np.concatenate([data["train_x"], data["val_x"], data["test_x"]], axis=0)
        y_vis = np.concatenate([data["train_y"], data["val_y"], data["test_y"]], axis=0)
        all_X.append(X_vis)
        all_y.append(to_cat(y_vis))

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    x_min, x_max = all_X[:, 0].min() - pad, all_X[:, 0].max() + pad
    y_min, y_max = all_X[:, 1].min() - pad, all_X[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points),
        np.linspace(y_min, y_max, grid_points),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    margin_maps = []

    for qbm in models:
        scores = []

        for point in grid:
            output_vals, _ = qbm.predict(point)
            scores.append(np.asarray(output_vals, dtype=float))

        scores = np.asarray(scores)

        margin = scores[:, 1] - scores[:, 0]
        margin_maps.append(margin.reshape(xx.shape))

    margin_maps = np.stack(margin_maps, axis=0)
    mean_margin = margin_maps.mean(axis=0)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, mean_margin, levels=12, cmap="RdBu_r", alpha=0.5)

    plt.contour(xx, yy, mean_margin, levels=[0.0], colors="k", linewidths=1.2, alpha=0.7)

    plt.scatter(
        all_X[:, 0], all_X[:, 1],
        c=all_y,
        cmap="RdBu_r",
        s=22
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    #plt.title(f"Mean decision boundary at epoch {best_epoch}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main(
    base_seed: int,
    new_run_path: str,
    run_name: str,
    chebyshev_degree: int,
    beta: float,
    batch_size: int,
    lr: float,
    gv_1: float,
    gv_2: float,
    n_seeds: int,
    best_epoch: int,
    use_moons: bool,
):
    os.makedirs(new_run_path, exist_ok=True)

    seeds = [478206, 538619, 168638, 215278, 243670, 124955, 947644, 537115, 853384, 267557]

    models = []
    all_data = []
    test_accs = []
    all_acc_lists = []
    all_loss_lists = []

    for seed in seeds:
        qbm, test_acc, acc_list, losses, data = train_one_seed(
            seed=seed,
            chebyshev_degree=chebyshev_degree,
            beta=beta,
            batch_size=batch_size,
            lr=lr,
            gv_1=gv_1,
            gv_2=gv_2,
            best_epoch=best_epoch,
            use_moons=use_moons,
        )
        models.append(qbm)
        all_data.append(data)
        test_accs.append(float(test_acc))
        all_acc_lists.append(acc_list)
        all_loss_lists.append(losses)

    test_accs = np.asarray(test_accs, dtype=float)
    acc_matrix = np.asarray(all_acc_lists, dtype=float)
    loss_matrix = np.asarray(all_loss_lists, dtype=float)

    summary = {
        "run_name": run_name,
        "n_epochs": int(best_epoch),
        "seeds": seeds,
        "acc_lists": all_acc_lists,
        "acc_mean_per_epoch": acc_matrix.mean(axis=0).tolist(),
        "acc_std_per_epoch": acc_matrix.std(axis=0).tolist(),

        # New Loss Data
        "loss_lists": all_loss_lists,
        "loss_mean_per_epoch": loss_matrix.mean(axis=0).tolist(),
        "loss_std_per_epoch": loss_matrix.std(axis=0).tolist(),

        "final_test_accs": test_accs.tolist(),
        "final_test_acc_mean": float(test_accs.mean()),
        "final_test_acc_std": float(test_accs.std()),
    }

    summary_path = os.path.join(new_run_path, f"{run_name}_summary.json")
    boundary_plot_path = os.path.join(new_run_path, f"{run_name}_decision_boundary_epoch{best_epoch:03d}.pdf")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    #plot_mean_decision_boundary(models, all_data, best_epoch, boundary_plot_path)

    print("\nSaved:")
    print(summary_path)
    print(boundary_plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed QBM with a fixed best epoch.")
    parser.add_argument("--seed", type=int, default=478206, help="Base seed")
    parser.add_argument("--path", type=str, default="out/rings/")
    parser.add_argument("--run_name", type=str, default="5h_fc_Z_ZZ")
    parser.add_argument("--chebyshev_degree", type=int, default=10)
    parser.add_argument("--beta", type=float, default=5.946560271525202)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.07358902393933116)
    parser.add_argument("--gv_1", type=float, default=1.0)
    parser.add_argument("--gv_2", type=float, default=1.0)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--best_epoch", type=int, default=20, help="Epoch chosen beforehand from validation")
    parser.add_argument("--use_moons", action="store_true", default=False)

    args = parser.parse_args()

    main(
        base_seed=args.seed,
        new_run_path=args.path,
        run_name=args.run_name,
        chebyshev_degree=args.chebyshev_degree,
        beta=args.beta,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        gv_1=args.gv_1,
        gv_2=args.gv_2,
        n_seeds=args.n_seeds,
        best_epoch=args.best_epoch,
        use_moons=args.use_moons,
    )