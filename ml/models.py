from copy import deepcopy
from typing import Any, Optional

import torch
from python_tools.ml.crf import CRF
from python_tools.ml.mixed import NeuralMixedEffects
from python_tools.ml.neural import MLP, LossModule, pop_with_prefix


class MLPCRF(LossModule):
    def __init__(self, *, mlp: bool = True, **kwargs):
        super().__init__(
            **{
                key: kwargs.pop(key)
                for key in ("sample_weight", "loss_function", "attenuation")
            }
        )
        if mlp:
            self.mlp = MLP(**kwargs)
        else:
            self.mlp = None
        self.crf = CRF(
            number_states=kwargs["output_size"], learn_sink=False, learn_source=False
        )

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.mlp is None:
            y_hat = x
        else:
            y_hat, meta = self.mlp(x, meta, y=y, dataset=dataset)
        meta["meta_scores"] = y_hat

        ids_index = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64),
                torch.nonzero(meta["meta_id"].diff(dim=0))[:, 0] + 1,
            ],
            dim=0,
        )

        # batch data
        assert torch.unique(meta["meta_id"]).shape[0] == ids_index.shape[0]
        data = []
        for iid, start in enumerate(ids_index):
            if iid < ids_index.shape[0] - 1:
                end = ids_index[iid + 1].item()
            else:
                end = y_hat.shape[0]
            data.append(y_hat[start:end])
        batched = torch.nn.utils.rnn.pad_sequence(
            data, batch_first=True, padding_value=-1.0
        )
        meta_crf = self.crf(batched, {}, dataset=dataset)[1]
        states = meta_crf["meta_states"]
        meta["meta_loss_scores"] = y_hat
        y_hat = torch.zeros_like(y_hat)
        y_hat[torch.arange(y_hat.shape[0]), states[states != -1]] = 1
        return y_hat, meta

    @torch.jit.export
    def loss(
        self,
        y_hat: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert take_mean
        assert loss is None
        ids_index = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64),
                torch.nonzero(meta["meta_id"].diff(dim=0))[:, 0] + 1,
            ],
            dim=0,
        )
        ys = []
        y_hats = []
        for iid, start in enumerate(ids_index):
            if iid < ids_index.shape[0] - 1:
                end = ids_index[iid + 1].item()
            else:
                end = y_hat.shape[0]
            ys.append(ground_truth[start:end].squeeze(1))
            y_hats.append(meta["meta_loss_scores"][start:end])
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1.0)
        y_hats = torch.nn.utils.rnn.pad_sequence(
            y_hats, batch_first=True, padding_value=-1.0
        )
        return self.crf.loss(y_hats, ys, meta)


class NMEWrapper(NeuralMixedEffects):
    def __init__(
        self,
        *,
        model_fun: type[LossModule] = MLP,
        fixed_model_fun: type[LossModule] | None = None,
        mixed_bias: bool = False,
        mixed_first: bool = False,
        mixed_all: bool = False,
        **kwargs: Any,
    ):
        """Rewrite kwargs and define fixed_model"""
        assert fixed_model_fun is None
        model_kwargs = pop_with_prefix(kwargs, "model_")
        fixed_model_fun = MLP
        if issubclass(model_fun, MLP) and (mixed_first or mixed_all):
            (
                fixed_model_kwargs,
                fixed_model_fun,
                model_kwargs,
                random_effects,
            ) = _first_all_mixed(mixed_first, {}, model_kwargs, ())
        elif issubclass(model_fun, MLP) or mixed_bias:
            # need to split last layer off
            if model_kwargs.get("layer_sizes", ()):
                # MAPS
                layers = model_kwargs["layer_sizes"]
                random_effects = ("layers.0.bias", "layers.0.weight")
            else:
                # TPOT
                layers = tuple(
                    x.bias.shape[0]
                    for x in MLP(
                        input_size=kwargs["input_size"],
                        output_size=kwargs["output_size"],
                        **model_kwargs,
                    ).layers
                    if isinstance(x, torch.nn.Linear)
                )[:-1]
                random_effects = ("layers.0.bias", "layers.0.weight")
            model_kwargs["layers"] = 0
            if not layers:
                # onyl one layer: remove fixed
                fixed_model_kwargs = {}
                fixed_model_fun = None
            else:
                fixed_model_kwargs = deepcopy(model_kwargs)
                # adjust layer, size, activation
                fixed_model_kwargs["final_activation"] = model_kwargs[
                    "activation"
                ].copy()
                fixed_model_kwargs["output_size"] = layers[-1]
                fixed_model_kwargs["layer_sizes"] = layers[:-1]
                model_kwargs["input_size"] = layers[-1]
            model_kwargs["layer_sizes"] = ()

            # combine linear and crf
            if mixed_bias:
                random_effects = tuple(f"mlp.{x}" for x in random_effects)
                if issubclass(model_fun, MLPCRF):
                    random_effects = (*random_effects, "crf.transition")
                else:
                    model_kwargs["alpha"] = fixed_model_kwargs.pop("alpha")
                    random_effects = (*random_effects, "alpha")

        elif issubclass(model_fun, MLPCRF):
            # need to move all paramters to fixed_model_
            fixed_model_kwargs = model_kwargs
            fixed_model_kwargs["output_size"] = kwargs["output_size"]
            model_kwargs = {"mlp": False}
            if issubclass(model_fun, MLPCRF):
                random_effects = ("crf.transition",)
            else:
                model_kwargs["alpha"] = fixed_model_kwargs.pop("alpha")
                random_effects = ("alpha",)

            if mixed_all or mixed_first:
                model_kwargs["mlp"] = True
                (
                    fixed_model_kwargs,
                    fixed_model_fun,
                    model_kwargs,
                    random_effects,
                ) = _first_all_mixed(
                    mixed_first, fixed_model_kwargs, model_kwargs, random_effects
                )

        else:
            assert False, model_fun

        fixed_model_kwargs = {
            f"fixed_model_{key}": value for key, value in fixed_model_kwargs.items()
        }
        model_kwargs = {f"model_{key}": value for key, value in model_kwargs.items()}

        super().__init__(
            model_fun=model_fun,
            fixed_model_fun=fixed_model_fun,
            random_effects=random_effects,
            independent=random_effects,
            **kwargs,
            **fixed_model_kwargs,
            **model_kwargs,
        )
        assert len(self.random) == len(random_effects)


def _first_all_mixed(
    mixed_first: bool,
    fixed_model_kwargs: dict,
    model_kwargs: dict,
    random_effects: tuple[str, ...],
) -> tuple[dict, None, dict, tuple[str, ...]]:
    # no fixed-only model
    model_kwargs.update(fixed_model_kwargs)

    prefix = ""
    if "mlp" in model_kwargs:
        prefix = "mlp."

    if mixed_first:
        random_effects = (
            *random_effects,
            f"{prefix}layers.0.bias",
            f"{prefix}layers.0.weight",
        )
    else:
        layers = (
            len(model_kwargs.get("layer_sizes", [])) + model_kwargs.get("layers", 0) + 1
        )
        random_effects = (
            *random_effects,
            *[f"{prefix}layers.{2*layer}.weight" for layer in range(layers)],
            *[f"{prefix}layers.{2*layer}.bias" for layer in range(layers)],
        )
    return {}, None, model_kwargs, random_effects
