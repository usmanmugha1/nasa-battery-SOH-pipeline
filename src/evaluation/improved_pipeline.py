# improved_pipeline.py
"""
Complete Improved Battery SOH Prediction Pipeline
- Improved D3R with proper trend/seasonal/noise decomposition
- ARIMA features for temporal dependencies
- TabPFN for superior prediction performance
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import BatteryDataLoader
from feature_engineering import BatteryFeatureEngineer
from ceemdan_decomposition import CEEMDANProcessor
from d3r_improved import ImprovedD3RProcessor
from arima_features import ARIMAFeatureEngineer
from tabpfn_predictor import BatterySOHPredictor

class ImprovedBatteryPipeline:
    """Complete improved pipeline with all enhancements"""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = BatteryDataLoader(data_dir)
        self.feature_engineer = BatteryFeatureEngineer()
        self.arima_engineer = ARIMAFeatureEngineer(max_lag=10, forecast_horizon=5)
        
    def process_single_battery(self, battery_id):
        """Process single battery through complete pipeline"""
        print(f"\n{'='*70}")
        print(f"PROCESSING BATTERY {battery_id}")
        print(f"{'='*70}")
        
        # Step 1: Load data
        print("\n[1/6] Loading and preprocessing data...")
        df = self.loader.process_battery(battery_id, self.output_dir)
        print(f"✓ Loaded {len(df)} cycles")
        
        # Step 2: Feature engineering
        print("\n[2/6] Engineering basic features...")
        df = self.feature_engineer.engineer_all_features(df)
        features_path = self.output_dir / f"B{battery_id:04d}_features.csv"
        df.to_csv(features_path, index=False)
        print(f"✓ Created {len(df.columns)} features")
        
        # Step 3: CEEMDAN decomposition
        print("\n[3/6] Applying CEEMDAN decomposition...")
        ceemdan = CEEMDANProcessor(trials=20, noise_std=0.05)
        ceemdan.decompose(df['capacity'])
        df_ceemdan = ceemdan.create_decomposed_features(df['capacity'])
        df = pd.concat([df, df_ceemdan], axis=1)
        
        ceemdan_path = self.output_dir / f"B{battery_id:04d}_ceemdan.csv"
        df.to_csv(ceemdan_path, index=False)
        
        # Plot CEEMDAN
        plot_path = self.output_dir / f"B{battery_id:04d}_ceemdan_plot.png"
        try:
            ceemdan.plot_decomposition(df['capacity'], save_path=plot_path)
        except Exception as e:
            print(f"⚠ CEEMDAN plot failed: {e}")
        
        print(f"✓ CEEMDAN complete")
        
        # Step 4: Improved D3R decomposition
        if len(df) >= 100:
            print("\n[4/6] Applying Improved D3R decomposition...")
            try:
                feature_cols = [
                    'capacity', 'voltage_mean', 'temperature_mean',
                    'discharge_time', 'resistance_proxy',
                    'capacity_rolling_mean_5', 'temp_rolling_mean_5',
                    'capacity_ewma', 'voltage_range', 'temp_increase'
                ]
                feature_cols = [f for f in feature_cols if f in df.columns]
                
                seq_len = min(50, len(df)//2)
                d3r = ImprovedD3RProcessor(seq_len=seq_len, n_features=len(feature_cols), d_model=128)
                
                # Train with more epochs for better decomposition
                d3r.train(df, feature_cols, epochs=150, batch_size=16, lr=0.001)
                
                # Decompose and plot
                d3r_plot_path = self.output_dir / f"B{battery_id:04d}_d3r_plot.png"
                d3r.plot_decomposition(df, feature_cols, save_path=d3r_plot_path)
                
                # Add D3R features
                df_d3r = d3r.decompose(df, feature_cols)
                df = pd.concat([df, df_d3r], axis=1)
                
                d3r_path = self.output_dir / f"B{battery_id:04d}_d3r.csv"
                df.to_csv(d3r_path, index=False)
                
                print(f"✓ Improved D3R complete")
            except Exception as e:
                print(f"⚠ D3R failed: {e}")
        else:
            print("\n[4/6] Skipping D3R (insufficient data)")
        
        # Step 5: ARIMA features
        print("\n[5/6] Engineering ARIMA features...")
        try:
            df = self.arima_engineer.engineer_all_arima_features(df, target_col='capacity')
            
            arima_path = self.output_dir / f"B{battery_id:04d}_arima.csv"
            df.to_csv(arima_path, index=False)
            
            print(f"✓ ARIMA features complete")
        except Exception as e:
            print(f"⚠ ARIMA features failed: {e}")
        
        # Step 6: Save final dataset
        print("\n[6/6] Saving final dataset...")
        final_path = self.output_dir / f"B{battery_id:04d}_final.csv"
        df.to_csv(final_path, index=False)
        
        print(f"\n✓ Battery {battery_id} processing complete!")
        print(f"  - Total cycles: {len(df)}")
        print(f"  - Total features: {len(df.columns)}")
        print(f"  - SOH range: {df['SOH'].min():.3f} - {df['SOH'].max():.3f}")
        print(f"  - Final data: {final_path}")
        
        return df
    
    def train_models_all_batteries(self, battery_ids=[5, 6, 7]):
        """Train models for all batteries"""
        print(f"\n{'='*70}")
        print("TRAINING TABPFN AND ENSEMBLE MODELS")
        print(f"{'='*70}")
        
        all_results = []
        
        for battery_id in battery_ids:
            print(f"\n{'='*70}")
            print(f"Battery {battery_id}")
            print(f"{'='*70}")
            
            # Load final data
            final_path = self.output_dir / f"B{battery_id:04d}_final.csv"
            
            if not final_path.exists():
                print(f"⚠ Data not found: {final_path}")
                continue
            
            df = pd.read_csv(final_path)
            
            # Train and evaluate
            try:
                predictor = BatterySOHPredictor(use_tabpfn=True)
                results, predictions, y_test, X_test = predictor.train_and_evaluate(
                    df, target_col='SOH', test_size=0.2
                )
                
                # Add battery ID to results
                results['Battery'] = f'B{battery_id:04d}'
                all_results.append(results)
                
                # Plot predictions
                plot_path = self.output_dir / f"B{battery_id:04d}_predictions.png"
                predictor.plot_predictions(y_test, predictions, battery_id=battery_id, save_path=plot_path)
                
                # Plot feature importance
                feature_cols = predictor.prepare_features(df)
                importance_path = self.output_dir / f"B{battery_id:04d}_feature_importance.png"
                predictor.plot_feature_importance(feature_cols, top_n=20, save_path=importance_path)
                
                print(f"\n✓ Battery {battery_id} modeling complete!")
                
            except Exception as e:
                print(f"⚠ Modeling failed for Battery {battery_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Combine all results
        if all_results:
            df_all_results = pd.concat(all_results, ignore_index=True)
            
            print(f"\n{'='*70}")
            print("FINAL SUMMARY - ALL BATTERIES")
            print(f"{'='*70}")
            print("\n" + df_all_results.to_string(index=False))
            
            # Save summary
            summary_path = self.output_dir / "final_summary_report.csv"
            df_all_results.to_csv(summary_path, index=False)
            print(f"\n✓ Summary saved: {summary_path}")
            
            # Print best models
            print(f"\n{'='*70}")
            print("BEST MODEL PER BATTERY")
            print(f"{'='*70}")
            for battery in df_all_results['Battery'].unique():
                battery_results = df_all_results[df_all_results['Battery'] == battery]
                best = battery_results.loc[battery_results['RMSE'].idxmin()]
                print(f"{battery}: {best['Model']} (RMSE: {best['RMSE']:.4f}, R²: {best['R²']:.4f})")
    
    def run_complete_pipeline(self, battery_ids=[5, 6, 7]):
        """Run complete pipeline for all batteries"""
        print("""
╔════════════════════════════════════════════════════════════════════╗
║           IMPROVED BATTERY SOH PREDICTION PIPELINE                 ║
║                                                                     ║
║  • Enhanced D3R with smooth trend extraction                       ║
║  • ARIMA features for temporal dependencies                        ║
║  • TabPFN for state-of-the-art predictions                        ║
╚════════════════════════════════════════════════════════════════════╝
        """)
        
        # Process each battery
        for battery_id in battery_ids:
            try:
                self.process_single_battery(battery_id)
            except Exception as e:
                print(f"\n❌ Error processing Battery {battery_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Train models
        self.train_models_all_batteries(battery_ids)
        
        print(f"\n{'='*70}")
        print("✓ COMPLETE PIPELINE FINISHED!")
        print(f"{'='*70}")
        print(f"All results saved to: {self.output_dir}")

# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/Users/binru/Development/ml_ai_project/5. Battery Data Set/1. BatteryAgingARC-FY08Q4"
    OUTPUT_DIR = "/Users/binru/Development/ml_ai_project/processed_data"
    
    # Run pipeline
    pipeline = ImprovedBatteryPipeline(DATA_DIR, OUTPUT_DIR)
    pipeline.run_complete_pipeline(battery_ids=[5, 6, 7])