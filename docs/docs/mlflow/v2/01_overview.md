# V2.0 - Hyperparameter Optimization with Optuna

!!! warn "Documentation Scope"
    This V2 documentation covers only the updated and newly created components. For the foundational logic and architecture structure, please refer to the V1 documentation.

## Project Goals for V2

The main goal of V2 is to **implement Optuna for automated hyperparameter discovery** across all model algorithms. While V1 established the core architecture using hardcoded hyperparameters, V2 seeks to unlock each algorithm's full potential through systematic parameter optimization.

## What is Optuna?

Optuna is an automatic hyperparameter optimization framework that uses intelligent sampling strategies to find optimal model parameters efficiently. Unlike grid search or random search, it focuses computational resources on promising parameter regions.

Optuna maintains optimization history through persistent storage, allowing studies to accumulate knowledge across multiple training sessions rather than starting from scratch each time.

**Official Resources:**

- Website: [https://optuna.org/](https://optuna.org/)
- Documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
- GitHub: [https://github.com/optuna/optuna](https://github.com/optuna/optuna)