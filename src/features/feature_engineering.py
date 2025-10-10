# feature_engineering.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

class BatteryFeatureEngineer:
    """Advanced feature engineering for battery SOH prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def add_rolling_features(self, df, windows=[3, 5, 10]):
        """Add rolling window statistics"""
        df = df.copy()
        
        for window in windows:
            # Rolling statistics for capacity
            df[f'capacity_rolling_mean_{window}'] = df['capacity'].rolling(window=window, min_periods=1).mean()
            df[f'capacity_rolling_std_{window}'] = df['capacity'].rolling(window=window, min_periods=1).std()
            df[f'capacity_rolling_min_{window}'] = df['capacity'].rolling(window=window, min_periods=1).min()
            df[f'capacity_rolling_max_{window}'] = df['capacity'].rolling(window=window, min_periods=1).max()
            
            # Rolling statistics for temperature
            df[f'temp_rolling_mean_{window}'] = df['temperature_mean'].rolling(window=window, min_periods=1).mean()
            df[f'temp_rolling_max_{window}'] = df['temperature_max'].rolling(window=window, min_periods=1).max()
        
        return df
    
    def add_degradation_features(self, df):
        """Add capacity degradation features"""
        df = df.copy()
        
        # Capacity fade rate
        df['capacity_diff'] = df['capacity'].diff()
        df['capacity_diff_pct'] = df['capacity'].pct_change()
        
        # Cumulative capacity fade
        initial_capacity = df['capacity'].iloc[0]
        df['cumulative_fade'] = initial_capacity - df['capacity']
        df['cumulative_fade_pct'] = (initial_capacity - df['capacity']) / initial_capacity
        
        # Acceleration of degradation (second derivative)
        df['capacity_diff_diff'] = df['capacity_diff'].diff()
        
        # Exponential weighted moving average
        df['capacity_ewma'] = df['capacity'].ewm(span=10, adjust=False).mean()
        
        return df
    
    def add_health_indicators(self, df):
        """Add health indicator features"""
        df = df.copy()
        
        # Voltage-based indicators
        df['voltage_range'] = df['voltage_max'] - df['voltage_min']
        df['voltage_variance'] = df['voltage_mean'].rolling(window=5, min_periods=1).var()
        
        # Temperature-based indicators
        df['temp_increase'] = df['temperature_max'] - df['ambient_temperature']
        
        # Discharge time trend
        df['discharge_time_change'] = df['discharge_time'].diff()
        
        # Internal resistance proxy (voltage drop / current)
        df['resistance_proxy'] = df['voltage_range'] / (df['current_mean'] + 1e-6)
        
        return df
    
    def add_cycle_features(self, df):
        """Add cycle-related features"""
        df = df.copy()
        
        # Cycle number features
        df['cycle_normalized'] = df['cycle'] / df['cycle'].max()
        df['cycle_squared'] = df['cycle'] ** 2
        df['cycle_sqrt'] = np.sqrt(df['cycle'])
        
        # Time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['days_elapsed'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / (24 * 3600)
        
        return df
    
    def add_statistical_features(self, df, window=10):
        """Add statistical features over sliding windows"""
        df = df.copy()
        
        # Skewness and kurtosis of capacity
        df['capacity_skew'] = df['capacity'].rolling(window=window, min_periods=1).apply(
            lambda x: stats.skew(x) if len(x) > 2 else 0)
        df['capacity_kurtosis'] = df['capacity'].rolling(window=window, min_periods=1).apply(
            lambda x: stats.kurtosis(x) if len(x) > 3 else 0)
        
        # Trend features
        def linear_trend(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        df['capacity_trend'] = df['capacity'].rolling(window=window, min_periods=2).apply(linear_trend)
        
        return df
    
    def add_regeneration_features(self, df):
        """Add features to capture capacity regeneration phenomenon"""
        df = df.copy()
        
        # Detect capacity increases (regeneration)
        df['capacity_increase'] = (df['capacity_diff'] > 0).astype(int)
        
        # Count regenerations in sliding window
        df['regeneration_count'] = df['capacity_increase'].rolling(window=10, min_periods=1).sum()
        
        # Variance in capacity changes (higher variance suggests regeneration)
        df['capacity_change_variance'] = df['capacity_diff'].rolling(window=10, min_periods=1).var()
        
        return df
    
    def engineer_all_features(self, df):
        """Apply all feature engineering steps"""
        print("Engineering features...")
        
        df = self.add_rolling_features(df)
        print("✓ Rolling features added")
        
        df = self.add_degradation_features(df)
        print("✓ Degradation features added")
        
        df = self.add_health_indicators(df)
        print("✓ Health indicators added")
        
        df = self.add_cycle_features(df)
        print("✓ Cycle features added")
        
        df = self.add_statistical_features(df)
        print("✓ Statistical features added")
        
        df = self.add_regeneration_features(df)
        print("✓ Regeneration features added")
        
        # Fill NaN values from differencing operations
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

# Example usage
if __name__ == "__main__":
    # Load processed data
    data_path = "/Users/binru/Development/ml_ai_project/processed_data/B0005_processed.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Engineer features
    engineer = BatteryFeatureEngineer()
    df_features = engineer.engineer_all_features(df)
    
    # Save
    output_path = data_path.replace('_processed.csv', '_features.csv')
    df_features.to_csv(output_path, index=False)
    print(f"\nFeature-engineered data saved to {output_path}")
    print(f"Total features: {len(df_features.columns)}")