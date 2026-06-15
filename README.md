# transplant-survival-model
1. Data Preparation
The model expects tabular features split into three categories:
Donor Features (e.g., AGE_DON, BMI_DON_CALC, ECD_DONOR)
Recipient Features (e.g., AGE, DIAL_VINT_YEARS, NUM_PREV_TX)
Match Features (e.g., HLAMIS, COLD_ISCH_KI)

Note: Medical datasets (like SRTR) are not included in this repository due to privacy restrictions. You must provide your own formatted .parquet or .csv dataset.

2. Training the Model
You can use the provided Jupyter notebooks (Train_1.ipynb, Train_2.ipynb) to step through the training process, or initialize the model directly in your scripts

3. Running Baselines & Cross-Validation
To compare CACRT against traditional models, you can utilize the cross-validation script to automatically train the baselines, run 5-fold CV, and generate boxplots comparing the C-index for both Graft Failure and DWFG.

# Model Architecture Highlights
Feature Tokenization: Converts scalar tabular features into d_model dimensional embeddings
Cross-Attention Streams: Learns how specific donor features interact with specific recipient features
Fusion & Risk Heads: Merges the attention streams with transplant match features to output Probability Mass Functions (PMF), Cumulative Incidence Functions (CIF), and overall survival probabilities

# Evaluation Metrics
The suite evaluates model performance using:
  * Concordance Index (C-index): Measures discriminative ability for specific competing risks
  * Time-Dependent AUC: Evaluates prediction accuracy at specific clinical horizons (e.g., 1-year, 5-year survival)
  * Integrated Brier Score (IBS): Assesses both calibration and discrimination over time
