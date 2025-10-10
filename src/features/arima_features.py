# arima_features.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

class ARIMAFeatureEngineer:
    """Create ARIMA-based features for time series prediction"""
    
    def __init__(self, max_lag=10, forecast_horizon=5):
        self.max_lag = max_lag
        self.forecast_horizon = forecast_horizon
        
    def create_lag_features(self, series, lags):
        """Create lagged features"""
        df_lags = pd.DataFrame()
        for lag in range(1, lags + 1):
            df_lags[f'lag_{lag}'] = series.shift(lag)
        return df_lags
    
    def create_rolling_stats(self, series, windows=[3, 5, 10]):
        """Create rolling statistics"""
        df_rolling = pd.DataFrame()
        for window in windows:
            df_rolling[f'rolling_mean_{window}'] = series.rolling(window, min_periods=1).mean()
            df_rolling[f'rolling_std_{window}'] = series.rolling(window, min_periods=1).std()
            df_rolling[f'rolling_min_{window}'] = series.rolling(window, min_periods=1).min()
            df_rolling[f'rolling_max_{window}'] = series.rolling(window, min_periods=1).max()
        return df_rolling
    
    def fit_arima_features(self, series, order=(2, 1, 2)):
        """Fit ARIMA and extract features"""
        try:
            model = ARIMA(series, order=order)
            fitted = model.fit()
            
            features = pd.DataFrame({
                'arima_residuals': fitted.resid,
                'arima_fitted': fitted.fittedvalues,
            })
            
            # Add forecasts as features
            for h in range(1, self.forecast_horizon + 1):
                forecast = fitted.forecast(steps=h)
                features[f'arima_forecast_h{h}'] = np.nan
                features.loc[features.index[-1], f'arima_forecast_h{h}'] = forecast.iloc[-1]
            
            # Forward fill forecasts
            for h in range(1, self.forecast_horizon + 1):
                features[f'arima_forecast_h{h}'] = features[f'arima_forecast_h{h}'].fillna(method='ffill')
            
            return features
            
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            return pd.DataFrame(index=series.index)
    
    def create_diff_features(self, series, orders=[1, 2]):
        """Create differencing features"""
        df_diff = pd.DataFrame()
        for order in orders:
            df_diff[f'diff_{order}'] = series.diff(order)
        return df_diff
    
    def create_autocorrelation_features(self, series, nlags=10):
        """Create autocorrelation features"""
        df_acf = pd.DataFrame(index=series.index)
        
        try:
            acf_values = acf(series.dropna(), nlags=nlags)
            pacf_values = pacf(series.dropna(), nlags=nlags)
            
            for lag in range(1, min(nlags + 1, len(acf_values))):
                df_acf[f'acf_{lag}'] = acf_values[lag]
                df_acf[f'pacf_{lag}'] = pacf_values[lag]
        except:
            pass
        
        return df_acf
    
    def engineer_all_arima_features(self, df, target_col='capacity'):
        """Create comprehensive ARIMA-based features"""
        print("Engineering ARIMA features...")
        
        series = df[target_col]
        
        # Lag features
        df_lags = self.create_lag_features(series, self.max_lag)
        print(f"✓ Created {len(df_lags.columns)} lag features")
        
        # Rolling statistics
        df_rolling = self.create_rolling_stats(series)
        print(f"✓ Created {len(df_rolling.columns)} rolling features")
        
        # Differencing
        df_diff = self.create_diff_features(series)
        print(f"✓ Created {len(df_diff.columns)} differencing features")
        
        # ARIMA features
        df_arima = self.fit_arima_features(series)
        print(f"✓ Created {len(df_arima.columns)} ARIMA features")
        
        # Autocorrelation (constant across all rows)
        df_acf = self.create_autocorrelation_features(series)
        print(f"✓ Created {len(df_acf.columns)} autocorrelation features")
        
        # Combine all
        df_combined = pd.concat([
            df,
            df_lags,
            df_rolling,
            df_diff,
            df_arima,
            df_acf
        ], axis=1)
        
        # Fill NaN
        df_combined = df_combined.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"✓ Total features: {len(df_combined.columns)}")
        
        return df_combined

# Test
if __name__ == "__main__":
    data_path = "/Users/binru/Development/ml_ai_project/processed_data/B0005_d3r_improved.csv"
    df = pd.read_csv(data_path)
    
    arima_engineer = ARIMAFeatureEngineer(max_lag=10, forecast_horizon=5)
    df_arima = arima_engineer.engineer_all_arima_features(df, target_col='capacity')
    
    output_path = data_path.replace('_d3r_improved.csv', '_arima.csv')
    df_arima.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")