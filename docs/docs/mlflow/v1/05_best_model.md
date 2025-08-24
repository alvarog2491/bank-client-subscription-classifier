# Reproduction Guide

## Complete Pipeline Execution

Steps to reproduce the V1.0 results.

## Prerequisites

Ensure MLFlow server is running and data is available in `data/raw/`.

## Execution Steps

```bash
# Run complete pipeline
mlflow run . -e main

# Or run individual steps
mlflow run . -e data_preprocessing
mlflow run . -e train -P model_type=lightgbm
```

Detailed reproduction instructions will be expanded in future versions.