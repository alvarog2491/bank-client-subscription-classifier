# Data Understanding

## Dataset Overview

- **Title**: Bank Term Deposit Subscription Prediction Dataset
- **Link**: https://www.kaggle.com/competitions/playground-series-s5e8/data
- **Source**: Portuguese banking institution marketing campaign data  
- **Original Dataset**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full)
- **Competition**: Kaggle Playground Series S5E8  

**Objective**: Predict whether a client will subscribe to a bank term deposit based on direct marketing campaign data.

The dataset originated from actual banking marketing campaigns but has been processed for the Kaggle competition. Each record represents a single client contacted during the marketing campaign.

## Data files

**train.csv** - training data with features and target  
**test.csv** - test features for final predictions  
**sample_submission.csv** - shows the submission format (id, probability)

## Feature Descriptions

The dataset contains 17 features representing client information and campaign details:

### Client Demographics
- **age**: Age of the client (numeric)
- **job**: Type of job (categorical: admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
- **marital**: Marital status (categorical: married, single, divorced)
- **education**: Education level (categorical: primary, secondary, tertiary, unknown)

### Financial Information
- **default**: Has credit in default? (categorical: yes, no)
- **balance**: Average yearly balance in euros (numeric)
- **housing**: Has housing loan? (categorical: yes, no)
- **loan**: Has personal loan? (categorical: yes, no)

### Campaign Contact Details
- **contact**: Communication type (categorical: unknown, telephone, cellular)
- **day**: Last contact day of month (numeric: 1-31)
- **month**: Last contact month (categorical: jan, feb, mar, ..., dec)
- **duration**: Last contact duration in seconds (numeric)

### Campaign History
- **campaign**: Number of contacts during this campaign (numeric)
- **pdays**: Days since last contact from previous campaign (numeric; -1 = not previously contacted)
- **previous**: Number of contacts before this campaign (numeric)
- **poutcome**: Outcome of previous campaign (categorical: unknown, other, failure, success)

### Target Variable
- **y**: Client subscribed to term deposit (binary: 1=yes, 0=no)
  
**→ See [Exploratory Data Analysis](eda.md) for detailed findings and visualizations.**

The analysis results guide model selection and preprocessing decisions for optimal prediction performance.