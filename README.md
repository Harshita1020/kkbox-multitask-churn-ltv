# KKBox Multi-Task MLP: Churn + LTV Prediction

A single PyTorch neural network that jointly predicts subscriber **churn**
and **lifetime value (LTV)** for KKBox Music Streaming (WSDM 2017 Kaggle).

## Results
| Metric | Value |
|--------|-------|
| Churn AUC-ROC | 0.844 |
| Churn AUC-PR | 0.356 (vs 0.064 base rate — 5.6x improvement) |
| LTV RMSE | 59.7 TWD |
| LTV R² | 0.484 |
| ECE (after calibration) | 0.001 |
| Revenue saved vs random | +769% |

## Architecture
- Factorization Machine interaction layer (pairwise feature crosses in O(k·d))
- Shared MLP backbone: 256 → 128 → 64 (BatchNorm + ReLU + Dropout)
- Dual output heads: BCE (churn) + MSE (LTV on log scale)
- 7 experiments: fixed weights, uncertainty weighting (Kendall 2018), PCGrad (Yu 2020)
- Isotonic regression probability calibration: ECE 0.265 → 0.001
- Business layer: Retention Priority Score = P(churn) × E[LTV]

## Project Structure
kkbox-multitask-churn-ltv/
├── config.py                     ← All paths and constants
├── models.py                     ← Dataset, FM layer, MultiTaskFMNet, PCGrad
├── utils.py                      ← Training loop, evaluation, ECE
├── 00_data_processing.py         ← ETL: .7z → typed Parquet
├── 01_eda.py                     ← EDA with DuckDB
├── 02_feature_engineering.py     ← Features, leakage fix, split
├── 04_training_baselines.py      ← Exp-1 and Exp-2
├── 05_multitask_ablation.py      ← Exp-3 to Exp-7
├── 06_calibration_business.py    ← Calibration + business layer
├── 07_final_evaluation.py        ← Final metrics and plots
├── models/                       ← Trained checkpoints
└── results/                      ← All plots and metrics

## How to Run
Data: kaggle.com/c/kkbox-churn-prediction-challenge

Run on Kaggle Notebooks (free GPU, data already available):
```bash
python 00_data_processing.py
python 02_feature_engineering.py
python 04_training_baselines.py
python 05_multitask_ablation.py
python 06_calibration_business.py
python 07_final_evaluation.py
```
