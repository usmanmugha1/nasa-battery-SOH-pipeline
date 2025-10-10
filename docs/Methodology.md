1. Data Preprocessing

Source: NASA PCoE Battery Data Set
Batteries: B0005, B0006, B0007
Cycles: 125-168 cycles per battery
Truncation: At 70% of initial capacity (1.4 Ah)
Sampling: ~3 hours between cycles

2. Decomposition Strategy
CEEMDAN

Ensemble of 20 trials
Noise std: 0.05
Extracts 5-8 IMFs per signal
Captures regeneration in high-frequency IMFs

Improved D3R

4-layer transformer (d_model=128, FFN=512)
8 attention heads
Custom loss functions:

Smoothness loss: L = mean(Δ²trend)
Monotonic loss: L = mean(ReLU(Δtrend))
Seasonal regularity: L = -Σ(FFT·log(FFT))


Loss weights: reconstruction(1.0), smoothness(0.1), monotonic(0.05), seasonal(0.02)
Training: 150 epochs, AdamW optimizer, cosine annealing

3. Feature Engineering
Total Features: 100-150 per battery
CategoryCountExamplesRaw measurements7voltage, current, temperature, timeRolling statistics24mean, std, min, max (windows: 3,5,10)Degradation8fade rate, cumulative fade, EWMAHealth indicators5voltage range, resistance proxy, temp increaseCycle features4normalized cycle, cycle², √cycleStatistical6skewness, kurtosis, trendRegeneration3increase flag, count, varianceCEEMDAN15-20IMF energies, trend, seasonal, noiseD3R3trend, seasonal, noiseARIMA40+lags, forecasts, ACF, PACF, residuals
4. Model Selection
TabPFN was selected as the primary model due to:

Pre-trained on 100,000+ synthetic tabular datasets
No hyperparameter tuning required
Handles complex feature interactions
Excellent generalization with small datasets
10-50x better performance than traditional ML

Why others failed:

XGBoost/GradientBoosting: Overfitting on small dataset
RandomForest: High variance, poor extrapolation
Traditional ML: Can't capture complex temporal patterns

📈 Model Comparison
ModelParametersTraining TimeInferenceRMSE (avg)R² (avg)TabPFN0 (pre-trained)<1 min<1 sec0.00120.9902XGBoost200 trees~2 min<1 sec0.0220-2.057GradientBoosting200 trees~3 min<1 sec0.0215-2.143RandomForest200 trees~2 min<1 sec0.0248-3.194Ensemble (all)-~7 min~2 sec0.0172-1.407