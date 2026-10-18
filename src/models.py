import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    LSTM-based regressor for RUL prediction.
    
    Input: (batch, 30, 14) - 30 time steps, 14 features
    Output: (batch,) - single RUL value per sample
    """
    def __init__(self, input_dim: int = 14, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last time step's output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.dropout(last_out)
        out = self.fc(out)  # (batch, 1)
        return out.squeeze(-1)  # (batch,)


if __name__ == "__main__":
    model = LSTMRegressor()
    x = torch.randn(32, 30, 14)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")