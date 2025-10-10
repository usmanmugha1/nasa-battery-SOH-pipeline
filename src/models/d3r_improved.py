# d3r_improved.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ImprovedSpatialTemporalTransformer(nn.Module):
    """Enhanced transformer with layer scaling and better attention"""
    
    def __init__(self, d_model, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        # Temporal attention with more heads for better pattern capture
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced feed-forward with residual
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # Better than ReLU for smooth gradients
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable scaling factors
        self.scale1 = nn.Parameter(torch.ones(1))
        self.scale2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Temporal attention with scaled residual
        attn_out, _ = self.temporal_attention(x, x, x)
        x = self.norm1(x + self.scale1 * self.dropout(attn_out))
        
        # Feed-forward with scaled residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.scale2 * self.dropout(ffn_out))
        
        return x

class ImprovedD3RDecomposer(nn.Module):
    """
    Improved D3R with:
    - Better trend extraction (smooth, monotonic)
    - Clearer seasonal patterns
    - Noise as residual
    """
    
    def __init__(self, n_features, d_model=128, n_layers=4, seq_len=50, dim_feedforward=512):
        super().__init__()
        
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input embedding with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )
        
        # Stacked transformer blocks (more layers for better decomposition)
        self.transformers = nn.ModuleList([
            ImprovedSpatialTemporalTransformer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=0.1
            )
            for _ in range(n_layers)
        ])
        
        # Separate decoders for each component with more capacity
        # Trend: should be smooth and monotonic
        self.trend_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Seasonal: should capture periodic patterns
        self.seasonal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Noise: residual component
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Apply transformers
        for transformer in self.transformers:
            x = transformer(x)
        
        # Decompose into components
        trend = self.trend_head(x).squeeze(-1)
        seasonal = self.seasonal_head(x).squeeze(-1)
        noise = self.noise_head(x).squeeze(-1)
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'embedding': x  # Return embeddings for analysis
        }

class ImprovedD3RProcessor:
    """Enhanced D3R processor with better training and decomposition"""
    
    def __init__(self, seq_len=50, n_features=10, d_model=128):
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_sequences(self, data):
        """Create overlapping sequences"""
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            seq = data[i:i+self.seq_len]
            sequences.append(seq)
        return np.array(sequences)
    
    def prepare_data(self, df, feature_cols):
        """Prepare and normalize data"""
        data = df[feature_cols].values
        data_scaled = self.scaler.fit_transform(data)
        sequences = self.create_sequences(data_scaled)
        return torch.FloatTensor(sequences).to(self.device)
    
    def smooth_trend_loss(self, trend):
        """Penalty for non-smooth trends"""
        # First derivative (velocity)
        first_diff = trend[:, 1:] - trend[:, :-1]
        # Second derivative (acceleration)
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        # Penalize large second derivatives (encourage smoothness)
        smoothness_loss = torch.mean(second_diff ** 2)
        return smoothness_loss
    
    def monotonic_trend_loss(self, trend):
        """Penalty for non-monotonic trends (battery should degrade)"""
        # Encourage negative slope (degradation)
        diffs = trend[:, 1:] - trend[:, :-1]
        # Penalize positive differences (capacity increasing)
        monotonic_loss = torch.mean(torch.relu(diffs))
        return monotonic_loss
    
    def seasonal_regularity_loss(self, seasonal):
        """Encourage periodic patterns in seasonal component"""
        # FFT to check if seasonal has dominant frequencies
        fft = torch.fft.rfft(seasonal, dim=1)
        fft_mag = torch.abs(fft)
        # Encourage concentration in few frequencies
        entropy = -torch.sum(fft_mag * torch.log(fft_mag + 1e-10), dim=1)
        regularity_loss = torch.mean(entropy)
        return regularity_loss
    
    def train(self, df, feature_cols, epochs=100, lr=0.001, batch_size=32):
        """Enhanced training with specialized losses"""
        print("Training Improved D3R model...")
        print(f"Device: {self.device}")
        
        X = self.prepare_data(df, feature_cols)
        print(f"Training data shape: {X.shape}")
        
        # Initialize model
        self.model = ImprovedD3RDecomposer(
            n_features=len(feature_cols),
            d_model=self.d_model,
            n_layers=4,
            seq_len=self.seq_len,
            dim_feedforward=512
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine annealing scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        mse_criterion = nn.MSELoss()
        
        # Data loader
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_smooth_loss = 0.0
            epoch_mono_loss = 0.0
            epoch_seasonal_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                batch_x = batch[0]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Reconstruction loss
                reconstructed = outputs['trend'] + outputs['seasonal'] + outputs['noise']
                target = batch_x[:, :, 0]  # Capacity is first feature
                recon_loss = mse_criterion(reconstructed, target)
                
                # Regularization losses
                smooth_loss = self.smooth_trend_loss(outputs['trend'])
                mono_loss = self.monotonic_trend_loss(outputs['trend'])
                seasonal_loss = self.seasonal_regularity_loss(outputs['seasonal'])
                
                # Combined loss with weights
                loss = (recon_loss + 
                       0.1 * smooth_loss + 
                       0.05 * mono_loss + 
                       0.02 * seasonal_loss)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_smooth_loss += smooth_loss.item()
                epoch_mono_loss += mono_loss.item()
                epoch_seasonal_loss += seasonal_loss.item()
                num_batches += 1
            
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Total Loss: {avg_loss:.6f}")
                print(f"  Recon: {epoch_recon_loss/num_batches:.6f}")
                print(f"  Smooth: {epoch_smooth_loss/num_batches:.6f}")
                print(f"  Mono: {epoch_mono_loss/num_batches:.6f}")
                print(f"  Seasonal: {epoch_seasonal_loss/num_batches:.6f}")
                print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("Training complete!")
    
    def decompose(self, df, feature_cols):
        """Decompose time series"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X = self.prepare_data(df, feature_cols)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            trend = outputs['trend'].cpu().numpy()
            seasonal = outputs['seasonal'].cpu().numpy()
            noise = outputs['noise'].cpu().numpy()
        
        # Use middle point of each sequence
        mid_idx = self.seq_len // 2
        trend_series = trend[:, mid_idx]
        seasonal_series = seasonal[:, mid_idx]
        noise_series = noise[:, mid_idx]
        
        # Pad to original length
        target_len = len(df)
        pad_len = target_len - len(trend_series)
        pad_start = pad_len // 2
        pad_end = pad_len - pad_start
        
        trend_padded = np.pad(trend_series, (pad_start, pad_end), mode='edge')
        seasonal_padded = np.pad(seasonal_series, (pad_start, pad_end), mode='edge')
        noise_padded = np.pad(noise_series, (pad_start, pad_end), mode='edge')
        
        # Additional smoothing for trend
        from scipy.ndimage import gaussian_filter1d
        trend_padded = gaussian_filter1d(trend_padded, sigma=2)
        
        return pd.DataFrame({
            'd3r_trend': trend_padded,
            'd3r_seasonal': seasonal_padded,
            'd3r_noise': noise_padded
        })
    
    def plot_decomposition(self, df, feature_cols, save_path=None):
        """Visualize decomposition quality"""
        df_decomposed = self.decompose(df, feature_cols)
        
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        
        # Original
        axes[0].plot(df['capacity'].values, 'b-', linewidth=2, label='Original')
        axes[0].set_ylabel('Capacity (Ah)', fontsize=10)
        axes[0].set_title('Original Capacity Signal', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(df_decomposed['d3r_trend'].values, 'r-', linewidth=2, label='Trend')
        axes[1].set_ylabel('Amplitude', fontsize=10)
        axes[1].set_title('D3R Trend (Degradation)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(df_decomposed['d3r_seasonal'].values, 'g-', linewidth=1.5, label='Seasonal')
        axes[2].set_ylabel('Amplitude', fontsize=10)
        axes[2].set_title('D3R Seasonal (Regeneration)', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        # Noise
        axes[3].plot(df_decomposed['d3r_noise'].values, 'orange', linewidth=1, alpha=0.7, label='Noise')
        axes[3].set_ylabel('Amplitude', fontsize=10)
        axes[3].set_title('D3R Noise (Residual)', fontsize=12, fontweight='bold')
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)
        
        # Reconstruction
        reconstructed = (df_decomposed['d3r_trend'].values + 
                        df_decomposed['d3r_seasonal'].values + 
                        df_decomposed['d3r_noise'].values)
        axes[4].plot(df['capacity'].values, 'b-', linewidth=2, label='Original', alpha=0.7)
        axes[4].plot(reconstructed, 'r--', linewidth=2, label='Reconstructed', alpha=0.7)
        axes[4].set_ylabel('Capacity (Ah)', fontsize=10)
        axes[4].set_xlabel('Cycle', fontsize=10)
        axes[4].set_title('Reconstruction Quality', fontsize=12, fontweight='bold')
        axes[4].legend(fontsize=9)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Decomposition plot saved: {save_path}")
        
        return fig

# Test
if __name__ == "__main__":
    data_path = "/Users/binru/Development/ml_ai_project/processed_data/B0005_ceemdan.csv"
    df = pd.read_csv(data_path)
    
    feature_cols = [
        'capacity', 'voltage_mean', 'temperature_mean',
        'discharge_time', 'resistance_proxy',
        'capacity_rolling_mean_5', 'temp_rolling_mean_5'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    d3r = ImprovedD3RProcessor(seq_len=50, n_features=len(feature_cols), d_model=128)
    d3r.train(df, feature_cols, epochs=100, batch_size=16, lr=0.001)
    
    df_d3r = d3r.decompose(df, feature_cols)
    df_combined = pd.concat([df, df_d3r], axis=1)
    
    output_path = data_path.replace('_ceemdan.csv', '_d3r_improved.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    plot_path = output_path.replace('.csv', '_plot.png')
    d3r.plot_decomposition(df, feature_cols, save_path=plot_path)