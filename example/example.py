from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from statsmodels.regression.mixed_linear_model import MixedLM

from python_tools.ml.mixed import NeuralMixedEffects
from python_tools.ml.neural import LossModule


class Model(LossModule):
    def __init__(
        self, *, input_size: int, output_size: int, loss_function="MSELoss"
    ) -> None:
        """A simple linear example model."""
        super().__init__(loss_function=loss_function)

        self.linear = torch.nn.Linear(input_size, output_size)

        # initializing from a fixed-only model can help with convergence
        """
        with torch.no_grad():
            self.linear.weight[:] = 0.5056
            self.linear.bias[:] = 0.0
        """

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: torch.Tensor | None = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_hat = self.linear(x)
        return y_hat, {"meta_y_hat": y_hat}


if __name__ == "__main__":
    # load data
    data = pd.read_csv("Orthodont.csv")
    data["Subject"] = pd.Categorical(data["Subject"]).codes
    data["age"] = (data["age"] - data["age"].mean()) / data["age"].std()
    data["distance"] = (data["distance"] - data["distance"].mean()) / data[
        "distance"
    ].std()
    data["intercept"] = 1
    data = data.astype(np.float32)

    # add mixed effects to pytorch Model (by default for all parameters)
    mixed = NeuralMixedEffects(
        # cluster names and the number of samples per cluster
        cluster_count=torch.from_numpy(
            data.groupby("Subject").count()["distance"].to_numpy()
        ),
        clusters=torch.from_numpy(data["Subject"].unique()),
        # the model
        model_fun=Model,
        # very important factor influencing the optimization process
        # this should be a hyper-parameter, for example, in [0.9, 1.0)
        # sensitive to learning rate and number of updates per epoch
        simulated_annealing_alpha=0.98,
        # arguments used by NeuralMixedEffects and forwarded to `model_fun`
        input_size=1,
        output_size=1,
        # which paramters have a mixed effects
        random_effects=("linear.weight", "linear.bias"),
    )

    # simple training loop
    x = torch.from_numpy(data.loc[:, ["age"]].to_numpy())
    meta = {"meta_id": torch.from_numpy(data.loc[:, ["Subject"]].to_numpy())}
    y = torch.from_numpy(data.loc[:, ["distance"]].to_numpy())
    optimizer = torch.optim.Adam(mixed.parameters(), lr=0.01)
    best_performance = float("inf")
    best_state = {}
    for epoch in range(200):
        # training step
        mixed.train()
        y_hat = mixed(x, meta=meta)[0]
        loss = mixed.loss(y, y_hat, meta=meta)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # evaluation step: internally updates the covariance matrix when self.training = False
        mixed.eval()
        performance = ((y - mixed(x, meta=meta)[0]) ** 2).mean()
        if performance < best_performance:
            best_performance = performance
            best_state = deepcopy(mixed.state_dict())

    mixed.load_state_dict(best_state)
    parameters = torch.cat(
        [
            (mixed.fixed["linear-weight"] + mixed.random["linear-weight"]).view(-1, 1),
            mixed.fixed["linear-bias"] + mixed.random["linear-bias"],
        ],
        dim=-1,
    )
    fixed = parameters.mean(dim=0)
    random = parameters - fixed
    covariance = random.T.cov()

    # test that paramters are within standard errors of statsmodel
    model = MixedLM(
        data["distance"],
        data.loc[:, ["age", "intercept"]],
        data["Subject"],
        data.loc[:, ["age", "intercept"]],
    ).fit()

    # variance of intercept/bias
    assert (
        covariance[1, 1] - model.cov_re.loc["intercept", "intercept"]
    ).abs() < model.bse_re["intercept Var"]

    # variance of age
    assert (covariance[0, 0] - model.cov_re.loc["age", "age"]).abs() < model.bse_re[
        "age Var"
    ]

    # covariance between intercept/bias and age
    assert (
        covariance[0, 1] - model.cov_re.loc["age", "intercept"]
    ).abs() < model.bse_re["age x intercept Cov"]

    # fixed: intercept/bias
    assert (fixed[1] - model.params["intercept"]).abs() < model.bse_fe["intercept"]

    # fixed: age
    assert (fixed[0] - model.params["age"]).abs() < model.bse_fe["age"]
