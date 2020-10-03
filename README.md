# Causai
Causai is a Python library for doing Causality in Machine Learning. We provide state-of-the-art causal & ML/DL algorithms into decision-making systems.

## Why Causai?
Solve the "Don't know what you don't know" step

There are alot of packages for each step causal inference. However, most of them assume the user has prior knowledge of domain-specific terminologies, many of which refer to the same concept/method. Also, different packages might have different API/processing pipeline, making it difficult to rapidly applying different methods to compare them. 

We want to provide a quick entrypoint to each step of causal inference by:
1. Conceptually categorizing different currently available methods and their implementations.
2. Unifying processing API of the most popular packages for each step, so the user can rapidly apply multiple methods and compare the results



## Main Features

Conceptual:
- Lower barrier to entry, giving the minimum background information to get started
- Help answer the question "Can I apply causal inference to my problem"?
- Categorize and compare different methods/packages
- Clearly provide short descriptions assumptions/caveats of each method and link to further resources.

Practical:
- Unify processing API from different packages, allowing user to apply multiple methods rapidly.
- Give a quick "baseline look" with sensible defaults while give users the option to tweak things based on domain knowledge.
- Perform several checks for assumptions of each method
- Based on the results, suggest interpretations and possible next step
- Meta-learning: Given the characteristics of your dataset, which method would be appropriate/promising to try?


# Get Started
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DanielhCarranza/causai/master)

## Installation 
```sh
    pip install causai
```

or 

```sh
git clone https://github.com/DanielhCarranza/causai.git
cd causai
```

## Example
```python
    import causai
    from causai.datasets import load_datasets
    from causai.inference import CausalInference
    from causai.discovery import AutoDiscovery

```





## Codebase

- **`causai`**
    - **`datasets`**: **Logic for downloading, preprocessing, augmenting, and loading data**
    - **`models`: Models wrap networks and add functionality like loss functions. saving, loading, and training**
    - **`networks` : Code for constructing ML model, neural net or bayesian net (dumb input | output mappings)**
    - **`tests`: Regression tests for the models code. Make sure a trained model performs well on important examples.**
    - **`metrics`**
    - **`estimator`**
    - `predictor.py`: **wrapper for model that allows you to do inference in the api**
    - `utils.py`

**`notebooks`:** **Examples, Tutorials, Explore and visualize data** 

**`tasks`** : **Scripts for running frequent tests and training commands**
**`training`**: **Logic for the training itself**
**`api`**: **Serve predictions. (Contains DockerFiles, Unit Tests, Flask, etc.)** 

**`evaluation`**: **Run the validation tests** 

**`experiment_manager`**: **Settings of your experiment manager (**p.e. wandb, tensorboard**)**

**`data`**: **use it for data versioning, storing data examples and metadata of your datasets. During training use it to store your raw and processed data but don't push or save the datasets into the repo.** 


