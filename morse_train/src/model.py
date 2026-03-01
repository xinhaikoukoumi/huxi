import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Identity()
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = z + residual
        z = self.act(z)
        return z


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        score = self.net(x).squeeze(-1)  # [B, T]
        weight = torch.softmax(score, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (x * weight).sum(dim=1)  # [B, C]


class MorseCharModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 26,
        input_channels: int = 2,
        variant: str = "enhanced_reslstm",
    ):
        super().__init__()
        self.variant = str(variant)
        if self.variant == "baseline_cnn_bilstm":
            self.features = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            )
            self.temporal = nn.LSTM(
                input_size=64,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = nn.Linear(128, num_classes)
        else:
            self.variant = "enhanced_reslstm"
            self.stem = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
            )
            self.block1 = ResidualBlock1D(32, 32, kernel_size=5, dropout=0.1)
            self.pool1 = nn.MaxPool1d(kernel_size=2)
            self.block2 = ResidualBlock1D(32, 64, kernel_size=5, dropout=0.1)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.block3 = ResidualBlock1D(64, 96, kernel_size=3, dropout=0.1)

            self.temporal = nn.LSTM(
                input_size=96,
                hidden_size=96,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attn_pool = AttentionPooling(in_dim=192, hidden_dim=96)
            self.head = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(192, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "baseline_cnn_bilstm":
            z = self.features(x)
            z = z.transpose(1, 2)
            z, _ = self.temporal(z)
            z = z.mean(dim=1)
            return self.classifier(z)

        z = self.stem(x)
        z = self.block1(z)
        z = self.pool1(z)
        z = self.block2(z)
        z = self.pool2(z)
        z = self.block3(z)
        z = z.transpose(1, 2)
        z, _ = self.temporal(z)
        z = self.attn_pool(z)
        return self.head(z)
