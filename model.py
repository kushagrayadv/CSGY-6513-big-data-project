import torch
import torch.nn as nn
import numpy as np

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=6, hidden=128, layers=3):
        super().__init__()
        self.enc  = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.bn   = nn.BatchNorm1d(hidden)
        self.act  = nn.LeakyReLU(0.2)
        self.dec  = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(hidden, input_size)

    def forward(self, x):
        b, t = x.shape[:2]
        _, (h, _) = self.enc(x)
        h_last = h[-1]
        h_seq  = h_last.unsqueeze(1).expand(b, t, -1)
        h_seq  = self.act(self.bn(h_seq.reshape(-1, h_seq.size(2)))).reshape(b, t, -1)
        out, _ = self.dec(h_seq)
        return self.proj(out)

    def predict(self, x, threshold):
        """
        Make prediction for a single window
        Args:
            x: Tensor of shape (1, seq_len, features)
            threshold: reconstruction error threshold for anomaly detection
        Returns:
            is_normal: boolean indicating if the sequence is normal
            recon_error: reconstruction error value
        """
        self.eval()
        with torch.no_grad():
            # Handle NaN values
            x = torch.nan_to_num(x, nan=0.0)
            
            # Get reconstruction
            recon = self(x)
            
            # Calculate reconstruction error (MSE)
            recon_error = torch.mean((recon - x) ** 2).item()
            
            # Compare with threshold
            is_normal = recon_error <= threshold
            
            return is_normal, recon_error 