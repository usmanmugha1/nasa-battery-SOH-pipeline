# Battery SOH Foundational Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art foundational model for battery State of Health (SOH) prediction using advanced time series decomposition and transformer-based methods.

## 🎯 Key Achievements

Our foundational model achieves **exceptional performance** on NASA battery degradation datasets:

| Battery | Model | RMSE | MAE | R² | MAPE |
|---------|-------|------|-----|-------|------|
| **B0005** | TabPFN | **0.0006** | 0.0004 | **0.9983** | 0.06% |
| **B0006** | TabPFN | **0.0013** | 0.0010 | **0.9971** | 0.13% |
| **B0007** | TabPFN | **0.0017** | 0.0013 | **0.9751** | 0.19% |

### Performance Comparison

Traditional ML models (XGBoost, RandomForest, GradientBoosting) achieved:
- RMSE: 0.013-0.033 (10-50x worse)
- R²: -4.5 to 0.79 (often negative)
- MAPE: 1.5-4.2%

**TabPFN consistently outperforms by 10-50x across all metrics.**

## 🌟 Features

### Advanced Decomposition Methods
- **CEEMDAN** (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)
  - Separates capacity signal into intrinsic mode functions (IMFs)
  - Captures regeneration phenomena and noise

- **Improved D3R** (Dynamic Decomposition with Diffusion Reconstruction)
  - Custom loss functions for smooth, monotonic trend extraction
  - Spatial-temporal transformer architecture
  - Specialized heads for trend, seasonal, and noise components
  - Regularization: smoothness loss, monotonic loss, seasonal regularity

### Temporal Feature Engineering
- **ARIMA-based features**
  - Lag features (1-10 lags)
  - Multi-horizon forecasts (1-5 steps ahead)
  - Autocorrelation and partial autocorrelation
  - Residuals and fitted values

- **Rolling statistics**
  - Multiple window sizes (3, 5, 10 cycles)
  - Mean, std, min, max, trend, skewness, kurtosis

- **Degradation indicators**
  - Capacity fade rate and acceleration
  - Cumulative degradation
  - Regeneration detection and counting
  - Internal resistance proxy

### State-of-the-Art Models
- **TabPFN**: Pre-trained transformer for tabular data (primary model)
- **Ensemble methods**: XGBoost, GradientBoosting, RandomForest
- **Automated model selection** and hyperparameter optimization


### **Total Features: 100-150 per battery**: 
Raw measurements (7) voltage, current, temperature, timeRolling 
statistics (24) mean, std, min, max (windows: 3,5,10)Degradation8fade rate, cumulative fade, EWMAHealth 
indicators (5) voltage range, resistance proxy, temp increase
Cycle features (4) normalized cycle, cycle², √cycle
Statistical (6) skewness, kurtosis, trend
Regeneration (3) increase flag, count, variance
CEEMDAN (15-20IMF) energies, trend, seasonal, noiseD3R3trend, seasonal, noise
ARIMA (40+) lags, forecasts, ACF, PACF, residuals

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────┐
│                    Raw Battery Data (.mat)                   │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│              Data Loading & Preprocessing                    │
│  • Extract discharge cycles                                  │
│  • Add timestamps                                            │
│  • Truncate at 70% capacity (EOL)                           │
│  • Calculate SOH                                             │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering (40+ features)              │
│  • Rolling statistics                                        │
│  • Degradation features                                      │
│  • Health indicators                                         │
│  • Statistical features                                      │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│              CEEMDAN Decomposition                           │
│  • Trend extraction                                          │
│  • Cyclical patterns                                         │
│  • Noise separation                                          │
│  • IMF energy features                                       │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│         Improved D3R Decomposition (Transformer)             │
│  • Smooth trend (degradation)                                │
│  • Seasonal patterns (regeneration)                          │
│  • Noise (residual)                                          │
│  • Embedding features                                        │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│              ARIMA Feature Engineering                       │
│  • Lag features                                              │
│  • Forecasts (1-5 steps)                                     │
│  • Autocorrelation                                           │
│  • Differencing                                              │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│            TabPFN + Ensemble Prediction                      │
│  • TabPFN (pre-trained transformer)                          │
│  • XGBoost, GradientBoosting, RandomForest                   │
│  • Ensemble averaging                                        │
│  • Feature importance analysis                               │
└────────────────────────┬────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   SOH Predictions                            │
│  • RMSE: 0.0006-0.0017                                       │
│  • R²: 0.975-0.998                                           │
│  • MAPE: 0.06-0.19%                                          │
└─────────────────────────────────────────────────────────────┘
