# tabpfn_predictor.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import TabPFN
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
    print("✓ TabPFN available")
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠ TabPFN not available. Install with: pip install tabpfn")

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")

class BatterySOHPredictor:
    """Enhanced SOH prediction with TabPFN and ensemble methods"""
    
    def __init__(self, use_tabpfn=True):
        self.use_tabpfn = use_tabpfn and TABPFN_AVAILABLE
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_features(self, df, target_col='SOH'):
        """Prepare features for modeling"""
        # Exclude non-feature columns
        exclude_cols = [
            'cycle', 'timestamp', 'type', 'SOH', 'capacity',
            'Unnamed: 0'  # Sometimes pandas adds this
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        # Remove columns with NaN or inf
        valid_cols = []
        for col in feature_cols:
            if not df[col].isna().any() and not np.isinf(df[col]).any():
                # Check for zero std (constant columns)
                if df[col].std() > 1e-10:
                    valid_cols.append(col)
        
        print(f"Selected {len(valid_cols)} valid features from {len(df.columns)} total columns")
        
        return valid_cols
    
    def resample_for_tabpfn(self, X, y, target_samples=1000):
        """Resample data to manageable size for TabPFN"""
        if len(X) <= target_samples:
            return X, y
        
        # Stratified resampling to preserve distribution
        indices = np.linspace(0, len(X) - 1, target_samples, dtype=int)
        return X[indices], y[indices]
    
    def train_tabpfn(self, X_train, y_train, X_test):
        """Train TabPFN model"""
        print("\nTraining TabPFN...")
        
        # Resample if needed (TabPFN works best with <1000 samples)
        X_train_resampled, y_train_resampled = self.resample_for_tabpfn(X_train, y_train, target_samples=1000)
        print(f"Resampled training data: {len(X_train_resampled)} samples")
        
        # TabPFN doesn't need scaling, but limit features
        max_features = min(100, X_train_resampled.shape[1])
        
        # Feature selection: use features with highest correlation to target
        correlations = []
        for i in range(X_train_resampled.shape[1]):
            corr = np.corrcoef(X_train_resampled[:, i], y_train_resampled)[0, 1]
            if np.isnan(corr):
                corr = 0
            correlations.append(np.abs(corr))
        
        correlations = np.array(correlations)
        top_features = np.argsort(correlations)[-max_features:]
        
        X_train_selected = X_train_resampled[:, top_features]
        X_test_selected = X_test[:, top_features]
        
        print(f"Using top {len(top_features)} features")
        
        # Initialize TabPFN with correct parameters
        try:
            # Try new API
            model = TabPFNRegressor(device='cpu')
        except TypeError:
            # Try old API
            try:
                model = TabPFNRegressor(device='cpu', N_ensemble_configurations=4)
            except:
                # Simplest initialization
                model = TabPFNRegressor()
        
        model.fit(X_train_selected, y_train_resampled)
        y_pred = model.predict(X_test_selected)
        
        self.models['TabPFN'] = {
            'model': model,
            'top_features': top_features
        }
        
        return y_pred
    
    def train_ensemble(self, X_train, y_train, X_test):
        """Train ensemble of models"""
        print("\nTraining ensemble models...")
        
        predictions = {}
        
        # Build models dictionary based on available packages
        models = {}
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            self.models[name] = {'model': model}
        
        return predictions
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE with protection against division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0
        
        return {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    def train_and_evaluate(self, df, target_col='SOH', test_size=0.2):
        """Complete training and evaluation pipeline"""
        print(f"\n{'='*60}")
        print("Training and Evaluating Models")
        print(f"{'='*60}")
        
        # Prepare features
        feature_cols = self.prepare_features(df, target_col)
        
        if len(feature_cols) < 5:
            raise ValueError(f"Insufficient features: {len(feature_cols)}")
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Remove any remaining NaN or inf
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X = X[mask]
        y = y[mask]
        
        print(f"Data shape after cleaning: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Replace any NaN that might have appeared during scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data (temporal split - no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        results = []
        all_predictions = {}
        
        # Train TabPFN if available
        if self.use_tabpfn:
            try:
                y_pred_tabpfn = self.train_tabpfn(X_train, y_train, X_test)
                all_predictions['TabPFN'] = y_pred_tabpfn
                results.append(self.evaluate_model(y_test, y_pred_tabpfn, 'TabPFN'))
            except Exception as e:
                print(f"⚠ TabPFN failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Train ensemble models
        ensemble_predictions = self.train_ensemble(X_train, y_train, X_test)
        
        for name, y_pred in ensemble_predictions.items():
            all_predictions[name] = y_pred
            results.append(self.evaluate_model(y_test, y_pred, name))
        
        # Create ensemble prediction (average of all models)
        if len(all_predictions) > 1:
            ensemble_pred = np.mean(list(all_predictions.values()), axis=0)
            all_predictions['Ensemble'] = ensemble_pred
            results.append(self.evaluate_model(y_test, ensemble_pred, 'Ensemble'))
        
        # Print results
        df_results = pd.DataFrame(results)
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(df_results.to_string(index=False))
        
        return df_results, all_predictions, y_test, feature_cols
    
    def plot_predictions(self, y_test, predictions, battery_id, save_path=None):
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: All predictions vs actual
        ax = axes[0]
        test_indices = range(len(y_test))
        ax.plot(test_indices, y_test, 'b-', label='Actual SOH', linewidth=2.5, alpha=0.8)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'cyan']
        linestyles = ['-', '--', '-.', ':', '-', '--']
        
        for idx, (name, y_pred) in enumerate(predictions.items()):
            ax.plot(test_indices, y_pred, 
                   label=f"{name}",
                   alpha=0.7, linewidth=1.5,
                   linestyle=linestyles[idx % len(linestyles)],
                   color=colors[idx % len(colors)])
        
        ax.set_xlabel('Test Sample Index', fontsize=11)
        ax.set_ylabel('SOH', fontsize=11)
        ax.set_title(f'Battery {battery_id}: SOH Predictions Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax = axes[1]
        for idx, (name, y_pred) in enumerate(predictions.items()):
            errors = y_test - y_pred
            ax.hist(errors, alpha=0.5, label=name, bins=20, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Prediction Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Battery {battery_id}: Error Distribution', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Prediction plot saved: {save_path}")
        
        plt.close()
        return fig
    
    def plot_feature_importance(self, feature_cols, top_n=20, save_path=None):
        """Plot feature importance from tree-based models"""
        # Try XGBoost first, then GradientBoosting
        model_key = 'XGBoost' if 'XGBoost' in self.models else 'GradientBoosting'
        
        if model_key not in self.models:
            print("⚠ No tree-based model available for feature importance")
            return
        
        model = self.models[model_key]['model']
        importances = model.feature_importances_
        
        # Get top features
        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        
        # Truncate long feature names
        feature_names = [feature_cols[i][:30] for i in indices]
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title(f'Top {top_n} Most Important Features ({model_key})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved: {save_path}")
        
        plt.close()
        return fig

# Test
if __name__ == "__main__":
    import os
    
    data_path = "/Users/binru/Development/ml_ai_project/processed_data/B0005_arima.csv"
    
    if not os.path.exists(data_path):
        # Try with final
        data_path = data_path.replace('_arima.csv', '_final.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found")
        exit(1)
    
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")
    
    predictor = BatterySOHPredictor(use_tabpfn=True)
    results, predictions, y_test, feature_cols = predictor.train_and_evaluate(df)
    
    # Plot predictions
    plot_path = data_path.replace('.csv', '_predictions.png')
    predictor.plot_predictions(y_test, predictions, battery_id='B0005', save_path=plot_path)
    
    # Plot feature importance
    importance_path = data_path.replace('.csv', '_importance.png')
    predictor.plot_feature_importance(feature_cols, top_n=20, save_path=importance_path)
    
    print(f"\n✓ All done!")