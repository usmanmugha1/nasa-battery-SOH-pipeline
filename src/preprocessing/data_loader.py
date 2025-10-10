# data_loader.py
import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class BatteryDataLoader:
    """Load and convert .mat battery data files to structured format"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.initial_capacity = 2.0  # Ah
        self.eol_threshold = 0.7  # 70% of initial capacity
        
    def load_mat_file(self, battery_id):
        """Load .mat file for specified battery"""
        file_path = self.data_dir / f"B{battery_id:04d}.mat"
        print(f"Loading {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        mat_data = scipy.io.loadmat(str(file_path))
        
        # Extract battery name (e.g., 'B0005')
        battery_key = [key for key in mat_data.keys() if key.startswith('B')][0]
        battery_data = mat_data[battery_key]
        
        return battery_data, battery_key
    
    def extract_cycle_data(self, battery_data, battery_id):
        """Extract and structure data from all cycles"""
        cycles_data = []
        
        # Get the cycle array
        cycles = battery_data['cycle'][0, 0]
        num_cycles = cycles.shape[1]
        
        print(f"Processing {num_cycles} cycles for battery {battery_id}")
        
        for cycle_idx in range(num_cycles):
            cycle = cycles[0, cycle_idx]
            
            # Extract cycle type
            cycle_type = str(cycle['type'][0]) if cycle['type'].size > 0 else ''
            
            # Extract ambient temperature
            amb_temp = float(cycle['ambient_temperature'][0, 0]) if cycle['ambient_temperature'].size > 0 else np.nan
            
            # Extract time (MATLAB date vector)
            time_vec = cycle['time'][0] if cycle['time'].size > 0 else np.array([])
            
            # Extract data based on cycle type
            if cycle_type == 'discharge' and cycle['data'].size > 0:
                data = cycle['data'][0, 0]
                
                # Extract discharge data
                voltage = data['Voltage_measured'][0] if 'Voltage_measured' in data.dtype.names else np.array([])
                current = data['Current_measured'][0] if 'Current_measured' in data.dtype.names else np.array([])
                temperature = data['Temperature_measured'][0] if 'Temperature_measured' in data.dtype.names else np.array([])
                time_series = data['Time'][0] if 'Time' in data.dtype.names else np.array([])
                capacity = data['Capacity'][0, 0] if 'Capacity' in data.dtype.names else np.nan
                
                cycle_info = {
                    'cycle': cycle_idx + 1,
                    'type': cycle_type,
                    'ambient_temperature': amb_temp,
                    'capacity': capacity,
                    'voltage_min': np.min(voltage) if len(voltage) > 0 else np.nan,
                    'voltage_max': np.max(voltage) if len(voltage) > 0 else np.nan,
                    'voltage_mean': np.mean(voltage) if len(voltage) > 0 else np.nan,
                    'voltage_std': np.std(voltage) if len(voltage) > 0 else np.nan,
                    'current_mean': np.mean(np.abs(current)) if len(current) > 0 else np.nan,
                    'current_std': np.std(current) if len(current) > 0 else np.nan,
                    'temperature_mean': np.mean(temperature) if len(temperature) > 0 else np.nan,
                    'temperature_max': np.max(temperature) if len(temperature) > 0 else np.nan,
                    'temperature_min': np.min(temperature) if len(temperature) > 0 else np.nan,
                    'temperature_std': np.std(temperature) if len(temperature) > 0 else np.nan,
                    'discharge_time': time_series[-1] if len(time_series) > 0 else np.nan,
                }
                
                cycles_data.append(cycle_info)
        
        return pd.DataFrame(cycles_data)
    
    def add_timestamps(self, df):
        """Add proper timestamps to the dataframe"""
        # Create timestamps assuming each cycle is approximately 3 hours apart
        base_time = datetime(2008, 4, 1)  # Arbitrary start date
        hours_per_cycle = 3
        
        df['timestamp'] = [base_time + timedelta(hours=i*hours_per_cycle) 
                          for i in range(len(df))]
        
        return df
    
    def truncate_at_eol(self, df):
        """Truncate data when capacity reaches 70% of initial"""
        eol_capacity = self.initial_capacity * self.eol_threshold
        
        if df['capacity'].min() < eol_capacity:
            eol_idx = df[df['capacity'] <= eol_capacity].index[0]
            print(f"Truncating at cycle {eol_idx + 1}, capacity: {df.loc[eol_idx, 'capacity']:.4f} Ah")
            df = df.loc[:eol_idx].copy()
        else:
            print(f"No truncation needed. Minimum capacity: {df['capacity'].min():.4f} Ah")
        
        return df
    
    def clean_data(self, df):
        """Clean and validate data"""
        # Remove any rows with NaN capacity
        df = df[df['capacity'].notna()].copy()
        
        # Ensure capacity is positive
        df = df[df['capacity'] > 0].copy()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def process_battery(self, battery_id, output_dir=None):
        """Complete processing pipeline for one battery"""
        # Load data
        battery_data, battery_key = self.load_mat_file(battery_id)
        
        # Extract cycle data
        df = self.extract_cycle_data(battery_data, battery_id)
        
        # Clean data
        df = self.clean_data(df)
        
        # Add timestamps
        df = self.add_timestamps(df)
        
        # Truncate at EOL
        df = self.truncate_at_eol(df)
        
        # Calculate SOH (State of Health)
        df['SOH'] = df['capacity'] / self.initial_capacity
        
        # Save to CSV if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            # Create directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_path = output_path / f"B{battery_id:04d}_processed.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved processed data to {file_path}")
        
        return df

# Example usage and testing
if __name__ == "__main__":
    # Set paths
    data_dir = "/Users/binru/Development/ml_ai_project/5. Battery Data Set/1. BatteryAgingARC-FY08Q4"
    output_dir = "/Users/binru/Development/ml_ai_project/processed_data"
    
    # Initialize loader
    loader = BatteryDataLoader(data_dir)
    
    # Process batteries 5, 6, 7
    for battery_id in [5, 6, 7]:
        try:
            print(f"\n{'='*50}")
            print(f"Processing Battery {battery_id}")
            print(f"{'='*50}")
            
            df = loader.process_battery(battery_id, output_dir)
            
            print(f"\n✓ Successfully processed Battery {battery_id}")
            print(f"  - Cycles processed: {len(df)}")
            print(f"  - SOH range: {df['SOH'].min():.3f} - {df['SOH'].max():.3f}")
            print(f"  - Capacity range: {df['capacity'].min():.3f} - {df['capacity'].max():.3f} Ah")
            print(f"  - Features: {len(df.columns)}")
            
            # Display first few rows
            print(f"\n  First 3 rows:")
            print(df.head(3).to_string())
            
            # Display last few rows
            print(f"\n  Last 3 rows:")
            print(df.tail(3).to_string())
            
        except Exception as e:
            print(f"\n✗ Error processing Battery {battery_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("Data loading complete!")
    print(f"{'='*50}")