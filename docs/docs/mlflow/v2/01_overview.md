# V2.0 - Hyperparameter Optimization with Optuna

!!! warn "Documentation Scope"
    This V2 documentation covers only the updated and newly created components. For the foundational logic and architecture structure, please refer to the V1 documentation.

## System Scope (V2.0)

The V2.0 implementation integrates **Optuna for automated hyperparameter discovery** across all model algorithms. While V1 established the core architecture using hardcoded hyperparameters, V2 unlocks each algorithm's potential through systematic parameter optimization.

## Optuna Integration

Optuna is an automatic hyperparameter optimization framework that uses intelligent sampling strategies to find optimal model parameters efficiently. Unlike grid search or random search, it focuses computational resources on promising parameter regions.

The system utilizes Optuna's persistent storage to maintain optimization history, allowing studies to accumulate knowledge across multiple training sessions.

**Official Resources:**

- Website: [https://optuna.org/](https://optuna.org/)
- Documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
- GitHub: [https://github.com/optuna/optuna](https://github.com/optuna/optuna)