"""
Four attention modules adapted from the thesis.
All operate on [batch, channels, seq_len] tensors.
"""

import math
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2020)."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        z = x.mean(dim=2)             # GAP → [B, C]
        z = self.sigmoid(self.fc2(self.relu(self.fc1(z))))  # [B, C]
        return x * z.unsqueeze(2)      # scale channels


class ECABlock(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020)."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1     # ensure odd
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        z = x.mean(dim=2)             # GAP → [B, C]
        z = z.unsqueeze(1)            # [B, 1, C]
        z = self.sigmoid(self.conv(z)) # [B, 1, C]
        z = z.squeeze(1).unsqueeze(2)  # [B, C, 1]
        return x * z


class TemporalBlock(nn.Module):
    """Temporal attention (DanHAR temporal branch, Gao et al., 2021)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        z_avg = x.mean(dim=1, keepdim=True)   # [B, 1, T]
        z_max = x.max(dim=1, keepdim=True)[0]  # [B, 1, T]
        z = torch.cat([z_avg, z_max], dim=1)   # [B, 2, T]
        z = self.sigmoid(self.conv(z))          # [B, 1, T]
        return x * z


class ChannelTemporalBlock(nn.Module):
    """Channel–temporal attention (DanHAR, Gao et al., 2021).
       Channel branch (CBAM-style) → temporal branch, in sequence.
    """

    def __init__(self, channels, reduction=2, temporal_kernel=7):
        super().__init__()
        # Channel branch (dual pooling + shared MLP)
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.ch_sigmoid = nn.Sigmoid()

        # Temporal branch
        pad = temporal_kernel // 2
        self.temp_conv = nn.Conv1d(2, 1, kernel_size=temporal_kernel,
                                   padding=pad, bias=False)
        self.temp_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ── Channel branch ──
        z_avg = x.mean(dim=2)                           # [B, C]
        z_max = x.max(dim=2)[0]                          # [B, C]
        ch_w = self.ch_sigmoid(
            self.fc2(self.relu(self.fc1(z_avg))) +
            self.fc2(self.relu(self.fc1(z_max)))
        )                                                 # [B, C]
        x = x * ch_w.unsqueeze(2)                        # scale channels

        # ── Temporal branch ──
        t_avg = x.mean(dim=1, keepdim=True)              # [B, 1, T]
        t_max = x.max(dim=1, keepdim=True)[0]            # [B, 1, T]
        t_cat = torch.cat([t_avg, t_max], dim=1)         # [B, 2, T]
        t_w = self.temp_sigmoid(self.temp_conv(t_cat))    # [B, 1, T]
        x = x * t_w                                       # scale time steps

        return x


def get_attention(name, channels, reduction=2, temporal_kernel=7):
    """Factory function to create attention modules by name."""
    if name == 'none':
        return nn.Identity()
    elif name == 'se':
        return SEBlock(channels, reduction)
    elif name == 'eca':
        return ECABlock(channels)
    elif name == 'temporal':
        return TemporalBlock(temporal_kernel)
    elif name == 'ct':
        return ChannelTemporalBlock(channels, reduction, temporal_kernel)
    else:
        raise ValueError(f"Unknown attention: {name}")
