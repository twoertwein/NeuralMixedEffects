#!/usr/bin/env python3
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from python_tools import caching
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.metrics import concat_dicts
from python_tools.ml.pytorch_tools import dict_to_batched_data

sys.path.append(".")


def within_split(data: pd.DataFrame, args) -> pd.Series:
    """Use the first 60% for training, 20% for valiudation, and 20% for testing."""
    training = (data["meta_id"] == data["meta_id"]) * 0
    for identifier in data["meta_id"].unique():
        index = data["meta_id"] == identifier
        if index.sum() < args.MIN_SAMPLES:
            # only for training
            if args.SMALLER > 0:
                # simulate smaller training dataset
                training[
                    training.index[index]
                    .to_series()
                    .sample(frac=args.SMALLER / 100, random_state=1)
                    .index
                ] = -1
            continue

        assert index.sum() >= args.MIN_SAMPLES
        current_data = data.loc[index, :]

        validation_indices = current_data.index[
            current_data["date"]
            >= current_data["date"].quantile(0.6, interpolation="lower")
        ]
        training.loc[validation_indices] = 1

        test_indices = current_data.index[
            current_data["date"]
            >= current_data["date"].quantile(0.8, interpolation="lower")
        ]
        training.loc[test_indices] = 2

        if args.SMALLER > 0:
            # simulate smaller training dataset
            training[
                training.index[index & (training == 0)]
                .to_series()
                .sample(frac=args.SMALLER / 100, random_state=1)
                .index
            ] = -1

    return training


def _get_partitions(data: pd.DataFrame, args) -> dict[int, dict[str, DataLoader]]:
    X_names = [x for x in data.columns if x not in ("meta_id", "date", "valence")]
    if args.LABEL not in ("valence", "positive", "negative"):
        data["date"] = (
            (data["date"].dt.strftime("%H%M%S%f").astype(float) / 1000)
            .round()
            .astype(int)
        )
    else:
        data["date"] = data["date"].dt.strftime("%y%m%d%H").astype(int)

    data.loc[:, X_names + ["valence"]] = data.loc[:, X_names + ["valence"]].astype(
        np.float32
    )

    y_names = [args.LABEL]
    if args.LABEL == "constructs":
        y_names = ["Other", "Aggressive", "Dysphoric", "Positive"]

    # define split
    splits = within_split(data, args)
    partition = {}
    for name, index in (
        ("training", splits == 0),
        ("validation", splits == 1),
        ("test", splits == 2),
    ):
        tmp = {
            "x": data.loc[index, X_names].values,
            "y": data.loc[index, "valence"].values,
            "meta_id": data.loc[index, "meta_id"].values,
            "meta_date": data.loc[index, "date"].values,
        }
        batched = dict_to_batched_data(tmp, batch_size=-1)
        if args.LABEL == "constructs":
            for batch in batched:
                batch["y"][0] = batch["y"][0].astype(int)

        partition[name] = DataLoader(
            batched,
            properties={
                "y_names": np.asarray(y_names),
                "x_names": np.asarray(X_names),
            },
        )

    return {0: partition}


def get_partitions(args):
    # load data
    if args.LABEL == "constructs":
        data = get_panam()
        args.FS = 55
        args.MIN_IQR = 0
        args.MIN_SAMPLES_TRAIN = 45
        args.MIN_SAMPLES = args.MIN_SAMPLES_TRAIN
    elif args.LABEL == "valence":
        args.MIN_IQR = 10
        args.MIN_SAMPLES = 50
        args.MIN_SAMPLES_TRAIN = 20
        data = pd.read_hdf("DAILY_features.hdf")
    elif args.LABEL.startswith("iemocap"):
        args.FS = 70
        args.MIN_SAMPLES = 50
        args.MIN_SAMPLES_TRAIN = 20
        args.MIN_IQR = 0.5
        data = get_iemocap(args.LABEL)
    else:
        data = get_lmmnn_dataset(args.LABEL)
        args.MIN_IQR = 0.01  # at least some variance
        args.MIN_SAMPLES = 20
        args.MIN_SAMPLES_TRAIN = 20
        args.FS = 70
        if args.LABEL == "spotify":
            args.FS = 100

    data = data.loc[~data["valence"].isna(), :]

    # at least n data points in total
    stats = data.groupby("meta_id")["valence"].describe()
    stats = stats.loc[
        (stats["count"] >= args.MIN_SAMPLES_TRAIN)
        & ((stats["75%"] - stats["25%"]) >= args.MIN_IQR)
    ]
    data = data.loc[data["meta_id"].isin(stats.index), :].copy()

    if args.MEAN_FREE:
        mean = data.loc[:, ["valence"]].groupby(data["meta_id"]).mean()
        for identifier in mean.index:
            index = data["meta_id"] == identifier
            data.loc[index, "valence"] -= mean.loc[identifier, "valence"]

    return _get_partitions(data, args)


def get_panam() -> pd.DataFrame:
    # convert to dataframe
    data, y_names = caching.read_pickle(Path("panam.pickle"))
    data = data[0]
    x_names = data["training"].property_dict["X_names"]
    for name in data:
        data[name] = concat_dicts(
            [
                {key: value[0] for key, value in batch.items()}
                for batch in data[name].iterator
            ]
        )
    data = concat_dicts(list(data.values()))
    data = pd.concat(
        [
            pd.DataFrame(
                {key: value.flatten() for key, value in data.items() if key != "X"}
            ),
            pd.DataFrame(data["X"], columns=x_names),
        ],
        axis=1,
    )
    day = datetime.datetime(year=2000, day=1, month=1)
    data["date"] = data.pop("meta_begin").apply(
        lambda x: day + datetime.timedelta(seconds=x)
    )
    return (
        data.sort_values(["meta_id", "date"], kind="stable")
        .drop(
            columns=[
                "meta_Evidence",
                "meta_Visual",
                "meta_Acoustic",
                "meta_Language",
                "meta_Interlocutor",
                "meta_end",
            ]
        )
        .rename(columns={"Y": "valence"})
        .reset_index(drop=True)
    )


def get_lmmnn_dataset(name: str):
    match name:
        case "news":
            y_name = "Facebook"
            id_name = "meta_source_id"
        case "spotify":
            y_name = "danceability"
            id_name = "meta_subgenre_id"
        case _:
            assert name == "imdb"
            y_name = "score"
            id_name = "meta_type_id"
    data = pd.read_csv(f"lmmnn_{name}_{y_name}.csv", index_col=0).rename(
        columns={"valence": "_valence"}
    )
    # fake date column for compatibility
    day = datetime.datetime(year=2000, day=1, month=1)
    data["date"] = [datetime.timedelta(seconds=i) + day for i in range(data.shape[0])]
    return data.rename(columns={y_name: "valence", id_name: "meta_id"})


def get_iemocap(name: str):
    assert name in ("iemocapa", "iemocapv")
    data = pd.read_csv("iemocap.csv").rename(columns={"meta_id": "meta_id_"})

    # within person (session+female)
    data["meta_id"] = (data["meta_session"] * 10 + data["meta_female"]).astype(int)

    # but scenario indepdent (impromt)
    day = datetime.datetime(year=2000, day=1, month=1)
    data["date"] = [
        datetime.timedelta(seconds=i) + day for i in data["meta_impro"].tolist()
    ]

    meta_name = "meta_Valence"
    if name == "iemocapa":
        meta_name = "meta_Arousal"
    return data.rename(columns={meta_name: "valence"})
