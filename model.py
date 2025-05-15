import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, input_size=6, cnn_ch=64, lstm_h=128, lstm_layers=2, n_classes=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_ch, 3, padding=1),
            nn.BatchNorm1d(cnn_ch), nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=1),
            nn.BatchNorm1d(cnn_ch), nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn_ch, lstm_h, lstm_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Linear(lstm_h*2, 1)
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(lstm_h*2, n_classes)

    def forward(self, x):                # x: (B,T,F)
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)    # (B,T,C)
        out,_ = self.lstm(x)                               # (B,T,2H)
        w = torch.softmax(self.attn(out).squeeze(-1), 1)   # (B,T)
        ctx = torch.sum(out * w.unsqueeze(-1), 1)          # (B,2H)
        return self.fc(self.drop(ctx))


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