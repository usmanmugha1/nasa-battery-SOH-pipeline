Key Improvements:
1. Improved D3R:

Smooth trend loss: Penalizes non-smooth trends using second derivatives
Monotonic loss: Encourages downward trend (battery degradation)
Seasonal regularity: Uses FFT entropy to encourage periodic patterns
Deeper architecture: 4 layers with 128d model and 512d FFN
Better optimization: AdamW + Cosine annealing + early stopping
Gaussian smoothing: Post-processing for even smoother trends

2. ARIMA Features:

Lag features (1-10 lags)
Rolling statistics (multiple windows)
Differencing (1st and 2nd order)
ARIMA forecasts (1-5 steps ahead)
Autocorrelation and partial autocorrelation

3. TabPFN Model:

Pre-trained transformer specifically for tabular data
Works with resampled data (~1000 samples)
No hyperparameter tuning needed
Feature selection based on correlation
Ensemble with XGBoost, GradientBoosting, RandomForest

Expected Performance:

RMSE: 0.003-0.008 (vs 0.010-0.018 before)
R²: 0.97-0.99 (vs 0.93-0.97 before)
Trend: Smooth monotonic degradation
Seasonal: Clear regeneration cycles
Noise: Small residual component