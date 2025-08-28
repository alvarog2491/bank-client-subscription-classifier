# Project Architecture

## Enhanced Base Model

V3 extends the existing factory pattern architecture with cross-validation capabilities in the `BaseModel` class.

!!! info "K-fold Integration"
    The `BaseModel` class now includes a `cross_validate()` method that performs stratified K-fold cross-validation. This method integrates seamlessly with the existing training pipeline while providing robust performance evaluation.

## Core Components

### Enhanced BaseModel Class
```python
class BaseModel(ABC):
    def cross_validate(self, X, y, cv_folds=5):
        """Perform stratified K-fold cross-validation"""
        # Creates StratifiedKFold with configured folds
        # Trains model on each fold
        # Returns mean  +-std for all metrics
```

### Configuration
Uses existing `config.yaml` structure with:

- `cv_folds: 5` for cross-validation configuration
- Existing model factory and hyperparameter settings