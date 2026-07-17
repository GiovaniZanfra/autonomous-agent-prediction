# Data Leakage Prevention Checklist

Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates during local validation and catastrophic failure on the private leaderboard.

When performing feature engineering, strictly adhere to the following principles:

## Target Leakage Prevention
- **Rule**: Ensure no feature is directly derived from or highly correlated with the target column in a way that would not be available at true inference time.
