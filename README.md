# Causai
Causai is a framework for doing Causality in Machine Learning. We provide state-of-the-art causal & ML algorithms into decision-making systems.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DanielhCarranza/causai/master)

## Codebase

**`notebooks`:** **Examples, Tutorials, Explore and visualize data** 

**`tasks`** : **Scripts for running frequent tests and training commands**

**`training`**: **Logic for the training itself**

- **`causai`**
    - **`datasets`**: **Logic for downloading, preprocessing, augmenting, and loading data**
    - **`models`: Models wrap networks and add functionality like loss functions. saving, loading, and training**
    - **`networks` : Code for constructing ML model, neural net or bayesian net (dumb input | output mappings)**
    - **`tests`: Regression tests for the models code. Make sure a trained model performs well on important examples.**
    - **`metrics`**
    - **`interpreter`**
    - `predictor.py`: **wrapper for model that allows you to do inference in the api**
    - `utils.py`

**`api`**: **Serve predictions. (Contains DockerFiles, Unit Tests, Flask, etc.)** 

**`evaluation`**: **Run the validation tests** 

**`experiment_manager`**: **Settings of your experiment manager (**p.e. wandb, tensorboard**)**

**`data`**: **use it for data versioning, storing data examples and metadata of your datasets. During training use it to store your raw and processed data but don't push or save the datasets into the repo.** 


