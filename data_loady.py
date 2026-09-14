from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo


@dataclass(frozen=True)
class DatasetSplit:
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray

    @property
    def num_features(self) -> int:
        return self.train_x.shape[1]

    def as_tuple(self):
        return (
            self.train_x,
            self.val_x,
            self.test_x,
            self.train_y,
            self.val_y,
            self.test_y,
        )


def split_dataset(X, y, seed: int) -> DatasetSplit:
    train_x, temp_x, train_y, temp_y = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x,
        temp_y,
        test_size=0.5,
        random_state=seed,
        stratify=temp_y,
    )
    return DatasetSplit(train_x, val_x, test_x, train_y, val_y, test_y)


def load_toy_data(seed: int, dataset: str = "circles") -> DatasetSplit:
    if dataset == "circles":
        X, y = make_circles(n_samples=1000, factor=0.1, noise=0.15, random_state=seed)
    elif dataset == "moons":
        X, y = make_moons(n_samples=1000, noise=0.15, random_state=seed)
    else:
        raise ValueError(f"Unknown toy dataset: {dataset}")

    X = X.astype(np.float32)
    y = y.astype(int).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X).astype(np.float32)
    X, y = sklearn.utils.shuffle(X, y, random_state=seed)

    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    return split_dataset(X, y_one_hot, seed)


def make_data(seed: int, use_moons: bool = False):
    dataset = "moons" if use_moons else "circles"
    return load_toy_data(seed, dataset=dataset).as_tuple()


def make_uci_data(seed: int):
    heart_disease = fetch_ucirepo(id=45)
    X_df = heart_disease.data.features.copy()
    y_df = heart_disease.data.targets.copy()

    df = X_df.copy()
    df["target"] = y_df

    df.dropna(inplace=True)

    if isinstance(df["target"], pd.DataFrame):
        df["target"] = df["target"].iloc[:, 0]

    df["target"] = df["target"].apply(lambda x: 1 if int(x) > 0 else 0).astype(int)

    categorical_features = ["cp", "restecg", "slope", "ca", "thal"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    binary_original_features = ["sex", "fbs", "exang"]
    for col in binary_original_features:
        df[col] = df[col].apply(lambda v: 1 if int(v) == 1 else 0).astype(int)

    for col in df.columns:
        if col == "target":
            continue

        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, False, True}):
            df[col] = df[col].astype(int) * 2 - 1

    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
    df[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])

    print("NaNs per column:\n", df.isnull().sum())
    print("Final input feature count:", df.drop("target", axis=1).shape[1])
    print("Feature ranges:")
    for col in df.drop("target", axis=1).columns:
        print(f"{col:12s} min={df[col].min(): .3f}, max={df[col].max(): .3f}")

    X = df.drop("target", axis=1).values.astype(np.float32)
    y_raw = df["target"].values.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    y = encoder.fit_transform(y_raw)

    return split_dataset(X, y, seed).as_tuple()


def load_dataset_split(seed: int, dataset: str = "circles") -> DatasetSplit:
    if dataset in {"circles", "moons"}:
        return load_toy_data(seed, dataset=dataset)
    if dataset == "uci":
        return DatasetSplit(*make_uci_data(seed))
    raise ValueError(f"Unknown dataset: {dataset}")
