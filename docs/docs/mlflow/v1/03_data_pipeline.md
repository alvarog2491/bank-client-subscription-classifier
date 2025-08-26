# Data Pipeline

## Current Implementation

The data pipeline consists of data loading and preprocessing stages.

!!! info "V1.0 Data Pipeline Scope"
    The dataset is already pre-cleaned and contains no missing values, so no additional cleaning or imputation is required. At this stage, the only preprocessing step is applying label encoding to categorical features to make them compatible with the models. 

## Data Loading

### Implementation
Located in `src/data/load_data.py`

### Functions Available
- `load_train_data()` - Loads training data from raw CSV
- `load_test_data()` - Loads test data from raw CSV  
- `load_data()` - Loads both training and test data
- `load_processed_data()` - Loads already processed data
- `load_label_encoders()` - Loads saved label encoder objects

### Data Loading Process
```python
# Load raw data
train_df, test_df = load_data()
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
```

The function prints data shapes and returns both datasets as pandas DataFrames.

## Data Preprocessing  

### Implementation
Located in `src/data/preprocess.py`

### MLFlow Entry Point
```bash
mlflow run . -e data_preprocessing
```

### Process Flow
The preprocessing performs label encoding on categorical features:

```python
def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply label encoding to categorical features and save encoders.
    
    Fits encoders on combined train+test data to ensure consistency.
    Saves processed datasets and encoders to configured paths.
    """
    config = load_config()

    # Load raw data
    train_df, test_df = load_data()
    
    # Get categorical features from config
    categorical_features = config["features"]["categorical_features"]
    
    # Process each categorical feature
    for feature in categorical_features:
        # Create label encoder
        le = LabelEncoder()
        
        # Fit on combined data for consistent encoding
        combined_values = pd.concat([
            train_df[feature].astype(str), 
            test_df[feature].astype(str)
        ]).unique()
        
        le.fit(combined_values)
        
        # Transform both datasets
        train_df[feature] = le.transform(train_df[feature].astype(str))
        test_df[feature] = le.transform(test_df[feature].astype(str))
```

### Output Files
```
data/processed/
├── train_processed.csv    # Processed training data
├── test_processed.csv     # Processed test data
└── label_encoders.pkl     # Saved label encoder objects
```


## Configuration

The data pipeline is controlled by settings in `config/config.yaml`:

```yaml
data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  train_file: "train.csv"
  test_file: "test.csv"
  target_column: "y"
  id_column: "id"

features:
  categorical_features: [job, marital, education, default, housing, loan, contact, month, poutcome]
  numerical_features: [age, balance, day, duration, campaign, pdays, previous]
  features_to_drop: [id]
  unknown_values: ["unknown"]
```

## Execution

### Run Preprocessing
```bash
# Via MLFlow entry point
mlflow run . -e data_preprocessing

# Direct execution
python -m src.data.preprocess
```

### Load Processed Data
```python
from src.data.load_data import load_processed_data, load_label_encoders

# Load processed datasets
train_df, test_df = load_processed_data()

# Load saved encoders for future use
encoders = load_label_encoders()
```
