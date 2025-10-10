# ceemdan_decomposition.py
import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
from scipy import signal

class CEEMDANProcessor:
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
    for battery capacity signal decomposition
    """
    
    def __init__(self, trials=100):
        self.ceemdan = CEEMDAN(trials=trials)
        self.imfs = None
        self.residue = None
        
    def decompose(self, capacity_series):
        """Decompose capacity signal using CEEMDAN"""
        print("Performing CEEMDAN decomposition...")
        
        # Ensure input is numpy array
        if isinstance(capacity_series, pd.Series):
            capacity_values = capacity_series.values
        else:
            capacity_values = np.array(capacity_series)
        
        # Perform decomposition
        self.imfs = self.ceemdan(capacity_values)
        self.residue = self.imfs[-1]
        
        print(f"Decomposition complete: {len(self.imfs)} IMFs extracted")
        
        return self.imfs
    
    def identify_components(self):
        """
        Identify different components:
        - High-frequency: noise and regeneration
        - Mid-frequency: cyclical patterns
        - Low-frequency: degradation trend
        """
        if self.imfs is None:
            raise ValueError("Must run decompose() first")
        
        n_imfs = len(self.imfs)
        
        components = {
            'high_freq': self.imfs[:max(1, n_imfs//3)],  # Noise/regeneration
            'mid_freq': self.imfs[max(1, n_imfs//3):2*n_imfs//3],  # Cycles
            'low_freq': self.imfs[2*n_imfs//3:],  # Trend
            'residue': self.residue
        }
        
        return components
    
    def reconstruct_trend(self):
        """Reconstruct the degradation trend (low-frequency components + residue)"""
        components = self.identify_components()
        
        # Sum low-frequency IMFs and residue
        trend = np.sum(components['low_freq'], axis=0)
        
        return trend
    
    def reconstruct_regeneration(self):
        """Reconstruct capacity regeneration pattern (high-frequency components)"""
        components = self.identify_components()
        
        # Sum high-frequency IMFs
        regeneration = np.sum(components['high_freq'], axis=0)
        
        return regeneration
    
    def plot_decomposition(self, capacity_series, save_path=None):
        """Visualize the decomposition results"""
        if self.imfs is None:
            raise ValueError("Must run decompose() first")
        
        n_imfs = len(self.imfs)
        fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2*n_imfs))
        
        # Plot original signal
        axes[0].plot(capacity_series, 'b', linewidth=1.5)
        axes[0].set_title('Original Capacity Signal', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Capacity (Ah)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot each IMF
        for i, imf in enumerate(self.imfs):
            axes[i+1].plot(imf, 'g', linewidth=1)
            axes[i+1].set_title(f'IMF {i+1}', fontsize=10)
            axes[i+1].set_ylabel('Amplitude')
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Cycle')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decomposition plot saved to {save_path}")
        
        return fig
    
    def create_decomposed_features(self, capacity_series):
        """Create features from decomposed components"""
        if self.imfs is None:
            self.decompose(capacity_series)
        
        # Get reconstructed components
        trend = self.reconstruct_trend()
        regeneration = self.reconstruct_regeneration()
        
        # Create DataFrame with decomposed features
        df_decomposed = pd.DataFrame({
            'capacity_trend': trend,
            'capacity_regeneration': regeneration,
            'capacity_cyclical': capacity_series.values - trend - regeneration
        })
        
        # Add energy of each IMF as features
        for i, imf in enumerate(self.imfs[:-1]):  # Exclude residue
            df_decomposed[f'imf_{i+1}_energy'] = imf ** 2
        
        return df_decomposed

# Example usage
if __name__ == "__main__":
    # Load data
    data_path = "/Users/binru/Development/ml_ai_project/processed_data/B0005_features.csv"
    df = pd.read_csv(data_path)
    
    # Apply CEEMDAN
    ceemdan = CEEMDANProcessor(trials=100)
    ceemdan.decompose(df['capacity'])
    
    # Create decomposed features
    df_decomposed = ceemdan.create_decomposed_features(df['capacity'])
    
    # Combine with original data
    df_combined = pd.concat([df, df_decomposed], axis=1)
    
    # Save
    output_path = data_path.replace('_features.csv', '_ceemdan.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"CEEMDAN features saved to {output_path}")
    
    # Plot
    plot_path = output_path.replace('.csv', '_plot.png')
    ceemdan.plot_decomposition(df['capacity'], save_path=plot_path)