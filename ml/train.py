#!/usr/bin/env python3
import argparse
import sys
from functools import partial
from pathlib import Path

from python_tools.generic import namespace_as_string
from python_tools.ml import metrics
from python_tools.ml.default.neural_models import GenericModel, MLPModel
from python_tools.ml.default.transformations import (
    DefaultTransformations,
    revert_transform,
    set_transform,
)
from python_tools.ml.evaluator import evaluator
from python_tools.ml.neural import MLP

sys.path.append(".")  # isort:skip

from data.loader import get_partitions
from ml.models import MLPCRF, NMEWrapper

EXCLUDE_NAMES = ("WORKERS",)
FOLDER = Path("evaluations/")


def combine_transformations(data, transform, model_transform=None):
    data = set_transform(data, transform)
    data.add_transform(model_transform, optimizable=True)
    return data


def train(partitions, args):
    workers = args.WORKERS
    y_names = tuple(partitions[0]["training"].properties["y_names"].tolist())

    metric = "mse"
    metric_max = False
    selection_metric = "mse"
    selection_metric_max = False
    metric_fun = partial(
        metrics.interval_metrics,
        names=y_names,
        which=("mse", "mae", "pearson", "ccc"),
    )

    if args.LABEL == "constructs":
        metric = "accuracy_proportional"
        metric_max = True
        selection_metric = "cohens_kappa_proportional"
        selection_metric_max = True
        metric_fun = partial(
            metrics.nominal_metrics,
            names=y_names,
            which=("accuracy", "cohens_kappa", "confusion"),
            clustering=True,
        )

    if args.NME or args.PER_PERSON:
        model = GenericModel(
            device="cuda",
            interval=args.LABEL != "constructs",
            nominal=args.LABEL == "constructs",
        )
    else:
        model = MLPModel(
            device="cuda",
            interval=args.LABEL != "constructs",
            nominal=args.LABEL == "constructs",
        )

    parameters = {
        "early_stop": [400],
        "epochs": [5_000],
        "lr": [0.01, 0.0001, 0.001, 0.005],
        "dropout": [0.0],
        "attenuation": [""],
        "sample_weight": [False],
        "minmax": [False],
        "final_activation": [{"name": "linear"}],
        "activation": [{"name": "ReLU"}],
        "layers": [0],
        "layer_sizes": [(2, 5), (5,), (5, 5), (10,), (10, 5)],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "loss_function": ["L1Loss", "MSELoss"],
        # for early stopping
        "metric": [metric],
        "metric_max": [metric_max],
    }
    if args.LABEL == "constructs":
        assert "linear" in args.LINK or "crf" in args.LINK or "all" in args.LINK
        parameters["layers"] = [0, 1, 2, 3]
        parameters["class_weight"] = [False]
        del parameters["layer_sizes"]
        del parameters["minmax"]
        parameters["loss_function"] = ["CrossEntropyLoss"]
    elif args.LABEL in ("news", "imdb", "spotify", "iemocapa", "iemocapv"):
        parameters["layers"] = [1, 2, 3]
        parameters["loss_function"] = ["MSELoss"]
        del parameters["layer_sizes"]
    elif "linear" in args.LINK:
        # only for MAPS
        parameters["layer_sizes"] = [(*sizes, 1) for sizes in parameters["layer_sizes"]]
        # for TPOT: "linear" = bias terms

    if args.NME or args.PER_PERSON:
        del model.parameters["final_activation"]
        # forward paramters
        forward = (
            "model_dropout",
            "model_final_activation",
            "model_activation",
            "model_layers",
            "model_layer_sizes",
        )
        for name in forward:
            base_name = name.removeprefix("model_")
            if base_name not in parameters:
                continue
            parameters[name] = parameters[base_name]
            del parameters[base_name]
        parameters["model_class"] = [NMEWrapper]
        parameters["model_fun"] = [MLP]
        if "crf" in args.LINK:
            parameters["model_fun"] = [MLPCRF]
        if "linear" in args.LINK and "+" in args.LINK:
            parameters["mixed_bias"] = [True]
            model.forward_names = (*model.forward_names, "mixed_bias")
        elif "first" in args.LINK:
            parameters["mixed_first"] = [True]
            model.forward_names = (*model.forward_names, "mixed_first")
        elif "all" in args.LINK:
            parameters["mixed_all"] = [True]
            model.forward_names = (*model.forward_names, "mixed_all")

        parameters["l2_lambda"] = [1.0]
        parameters["simulated_annealing_alpha"] = [0.97, 0.99, 0.95, 0.93]
        if args.PER_PERSON:
            parameters["l2_lambda"] = [0.0]
            parameters["simulated_annealing_alpha"] = [0.97]  # not used
        if args.IND:
            # no fixed effects
            parameters["exclude_parameters_prefixes"] = [(("fixed",),)]
        model.forward_names = (
            *model.forward_names,
            *forward,
            "model_fun",
            "random_effects",
            "independent",
            "clusters",
            "cluster_count",
            "l2_lambda",
            "simulated_annealing_alpha",
        )
        model.static_kwargs = (*model.static_kwargs, "final_activation")

    elif "crf" in args.LINK:
        parameters["model_class"] = [MLPCRF]

    elif args.LME:
        assert args.LABEL != "constructs"
        parameters["final_activation"] = [{"name": "lme", "iterations": 14}]
        if args.LABEL == "valence":
            parameters["final_activation"].append(
                {"only_bias": True, **parameters["final_activation"][0]}
            )

    model.parameters.update(parameters)
    models, parameters, transform_ = model.get_models()

    # transformations
    transform = DefaultTransformations(
        interval=args.LABEL != "constructs", nominal=args.LABEL == "constructs"
    )
    apply_transformation = partial(combine_transformations, model_transform=transform_)
    models = (models[0],) * len(parameters)
    transform_parameter = ({"feature_selection": f"univariate_{args.FS}"},) * len(
        partitions
    )

    folder = FOLDER / namespace_as_string(args, exclude=EXCLUDE_NAMES)
    print(folder, len(models))
    evaluator(
        models=models,
        partitions=partitions,
        workers=workers,
        parameters=parameters,
        folder=folder,
        metric_fun=metric_fun,
        learn_transform=transform.define_transform,
        apply_transform=apply_transformation,
        revert_transform=revert_transform,
        transform_parameter=transform_parameter,
        concat_folds=True,
        # for model selection
        metric=selection_metric,
        metric_max=selection_metric_max,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--FS", type=int, default=70)
    parser.add_argument("--SMALLER", type=int, default=0)
    parser.add_argument(
        "--LINK",
        type=str,
        default="linear",
        choices=["linear", "crf", "first", "all", "linear+crf", "first+crf", "all+crf"],
    )
    parser.add_argument("--IND", action="store_const", const=True, default=False)
    parser.add_argument("--NME", action="store_const", const=True, default=False)
    parser.add_argument("--LME", action="store_const", const=True, default=False)
    parser.add_argument("--PER_PERSON", action="store_const", const=True, default=False)
    parser.add_argument("--WORKERS", type=int, default=8)
    parser.add_argument("--MIN_SAMPLES", type=int, default=50)
    parser.add_argument("--MIN_SAMPLES_TRAIN", type=int, default=20)
    parser.add_argument("--MIN_IQR", type=int, default=10)
    parser.add_argument(
        "--LABEL",
        type=str,
        default="valence",
        choices=[
            "valence",
            "positive",
            "negative",
            "constructs",
            "imdb",
            "news",
            "spotify",
            "iemocapv",
            "iemocapa",
        ],
    )
    parser.add_argument("--MEAN_FREE", action="store_const", const=True, default=False)

    args = parser.parse_args()

    if ("all" in args.LINK or "first" in args.LINK) and not (
        args.NME or args.PER_PERSON
    ):
        sys.exit(0)

    train(get_partitions(args), args)
