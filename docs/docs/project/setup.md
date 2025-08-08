# Setup Guide

This guide will help you set up the complete environment for the Bank Client Subscription Classifier project.

## Prerequisites

- Python 3.9 or higher
- Git
- 8GB+ RAM recommended
- 2GB+ free disk space

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd bank-client-subscription-classifier
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## MLFlow Setup

### 1. Initialize MLFlow

```bash
# Create MLFlow directories
mkdir -p mlruns mlartifacts

# Start MLFlow server (in a separate terminal)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000
```

### 2. Verify MLFlow Installation

Visit [http://localhost:5000](http://localhost:5000) to access the MLFlow UI.

## Data Setup

### 1. Download Competition Data

1. Go to [Kaggle Competition Page](https://kaggle.com/competitions/playground-series-s5e8)
2. Download the dataset files:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

### 2. Place Data Files

```bash
# Create data directories
mkdir -p data/raw data/processed data/external

# Place downloaded files
cp path/to/downloaded/train.csv data/raw/
cp path/to/downloaded/test.csv data/raw/
cp path/to/downloaded/sample_submission.csv data/raw/
```

## Documentation Setup

### 1. Build Documentation

```bash
# Serve documentation locally
mkdocs serve

# Visit http://localhost:8000 to view docs
```

### 2. Build Static Site

```bash
# Build static documentation
mkdocs build

# Documentation will be in site/ directory
```

## Development Tools Setup

### 1. Code Formatting

```bash
# Format code
black src/ experiments/ tests/

# Check code style
flake8 src/ experiments/ tests/
```

### 2. Testing

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/ --cov-report=html
```

## Configuration

### 1. Update Configuration

Edit `config/config.yaml` to match your setup:

```yaml
project:
  name: "bank-client-subscription-classifier"
  author: "Álvaro Granados"

mlflow:
  tracking_uri: "http://localhost:5000"
```

## Verification

### 1. Quick Test Run

```bash
# Test data loading
python -c "from src.data.load_data import *; print('Data module works!')"

# Test MLFlow connection
python -c "import mlflow; print('MLFlow version:', mlflow.version.VERSION)"
```

### 2. Run Sample Experiment

```bash
# Run a simple experiment
mlflow run . -e main
```

## IDE Setup

### Visual Studio Code

Recommended extensions:
- Python
- Jupyter
- MLFlow
- YAML
- Markdown All in One

### JetBrains PyCharm

Configure:
- Python interpreter: `./venv/bin/python`
- Project structure: Mark `src/` as source root

## Troubleshooting

### Common Issues

**MLFlow server won't start:**
```bash
# Check if port is in use
lsof -i :5000

# Use different port
mlflow server --port 5001 ...
```

**Import errors:**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

**Permission errors:**
```bash
# Fix directory permissions
chmod -R 755 mlruns/ mlartifacts/
```

## Next Steps

After setup is complete:

1. 📊 **Explore Data**: Open `notebooks/01_exploratory_data_analysis.ipynb`
2. 🛠️ **Build Features**: Start with `src/features/build_features.py`
3. 🤖 **Train Models**: Use `src/models/train_model.py`
4. 📈 **Track Experiments**: Monitor in MLFlow UI
5. 📚 **Document Progress**: Update MKDocs as you go

Happy coding! 🚀