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
