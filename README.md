# Neural Mixed Effect Models

This repository contains the evaluation code for ["Neural Mixed Effects for Nonlinear Personalized Predictions"](https://arxiv.org/abs/2306.08149) which is accepted at [ICMI'23](https://icmi.acm.org/2023/).

[Please see here](https://bitbucket.org/twoertwein/python-tools/src/879e7937313ff63c86fccae9b02fc5151e7ac069/python_tools/ml/mixed.py#lines-503), if you are looking for only the Neural Mixed Effects (NME) code.

## Setup
```sh
git clone git@github.com:twoertwein/NeuralMixedEffects.git
cd NeuralMixedEffects
poetry update  # installs dependencies
```

## Usage
```sh
# NN-Generic
poetry run python ml/train.py --LINK <link> --LABEL <label>
# NN-Specific
poetry run python ml/train.py --LINK <link> --LABEL <label> --PER_PERSON --IND
# MLP-LME
poetry run python ml/train.py --LINK <link> --LABEL <label> --LME
# unregularized NME
poetry run python ml/train.py --LINK <link> --LABEL <label> --PER_PERSON
# NME
poetry run python ml/train.py --LINK <link> --LABEL <label> --NME
```

Where `<label>` can be: `imdb`, `news`, `spotify`, `iemocapa` (arousal on IEMOCAP), `iemocapv` (valence on IEMOCAP), `valence` (daily mood on MAPS), and `constructs` (the four affective states on TPOT).

`<link>` determines the model type where person-specific parameters are. It can be `linear` (last layer), `all` (all layers), `first` (first layer), `crf` (transition matrix T of CRF), `linear+crf` (last MLP layer and T), `first+crf` (first MLP layer and T), and `all+crf` (all MLP layers and T).

To reduce the training data for each person, please use `--SMALLER 20` to train 20% less data per person.

## Data

The features for Imdb, News, Spotify, MAPS, and TPOT are available [here](https://cmu.box.com/s/7d376xsqccw5evbx4n8wyxd9cv08teyc). If you want the features for [IEMOCAP](https://cmu.box.com/shared/static/w8yn9a7467onw7wfwgsszxcrk0fzve9m.csv), please send us proof that you completed the data-sharing agreement required by IEMOCAP.
