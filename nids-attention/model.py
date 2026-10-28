"""
Depthwise-separable 1D-CNN for network intrusion detection.
Mirrors the thesis architecture with configurable attention.
"""

import torch
import torch.nn as nn
from attention import get_attention


class DSConvBlock(nn.Module):
    """One depthwise-separable conv block with optional attention."""

    def __init__(self, in_ch, out_ch, kernel_size, attention_name='none',
                 reduction=2, temporal_kernel=7, dropout=0.1):
        super().__init__()
        # Depthwise
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            padding=kernel_size // 2, groups=in_ch, bias=False)
        # Pointwise
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU(inplace=True)
        # Attention (after ReLU, before Dropout — same as thesis)
        self.attn = get_attention(attention_name, out_ch, reduction, temporal_kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.attn(x)     # ← attention here
        x = self.drop(x)
        return x


class NIDSNet(nn.Module):
    """
    1D-CNN for network intrusion detection.

    Architecture mirrors the thesis:
      - Initial Conv1d to expand input channels
      - N depthwise-separable blocks with attention
      - Global average pooling
      - Linear classifier

    Args:
        in_channels:     number of input channels (1 or n_groups)
        seq_len:         length of the input sequence (n_features or group_size)
        n_classes:       number of output classes
        n_blocks:        number of DS-conv blocks
        channels:        channel width after first conv
        kernel_size:     convolution kernel size
        attention:       'none', 'se', 'eca', 'temporal', 'ct'
        reduction:       SE/CT reduction ratio
        temporal_kernel:  temporal attention kernel size
        dropout:         dropout rate
    """

    def __init__(self, in_channels=1, seq_len=78, n_classes=8,
                 n_blocks=4, channels=32, kernel_size=9,
                 attention='none', reduction=2, temporal_kernel=7,
                 dropout=0.1):
        super().__init__()

        # Initial projection: in_channels → channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # DS-conv blocks
        self.blocks = nn.ModuleList([
            DSConvBlock(channels, channels, kernel_size,
                        attention, reduction, temporal_kernel, dropout)
            for _ in range(n_blocks)
        ])

        # Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(2)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config):
    """Build model from a config dict."""
    model = NIDSNet(
        in_channels=config.get('in_channels', 1),
        seq_len=config.get('seq_len', 78),
        n_classes=config.get('n_classes', 8),
        n_blocks=config.get('n_blocks', 4),
        channels=config.get('channels', 32),
        kernel_size=config.get('kernel_size', 9),
        attention=config.get('attention', 'none'),
        reduction=config.get('reduction', 2),
        temporal_kernel=config.get('temporal_kernel', 7),
        dropout=config.get('dropout', 0.1),
    )
    return model


if __name__ == '__main__':
    # Quick sanity check
    configs = [
        {'attention': 'none',     'label': 'Baseline'},
        {'attention': 'se',       'label': 'SE (r=2)'},
        {'attention': 'eca',      'label': 'ECA'},
        {'attention': 'temporal', 'label': 'Temporal'},
        {'attention': 'ct',       'label': 'CT (r=2)'},
    ]
    print(f"{'Variant':<20} {'Params':>10}")
    print("-" * 32)
    for cfg in configs:
        m = build_model({**cfg, 'in_channels': 1, 'seq_len': 78, 'n_classes': 8})
        print(f"{cfg['label']:<20} {m.count_parameters():>10,}")

    # Forward pass test
    x = torch.randn(4, 1, 78)
    for cfg in configs:
        m = build_model({**cfg, 'in_channels': 1, 'seq_len': 78, 'n_classes': 8})
        out = m(x)
        assert out.shape == (4, 8), f"Failed for {cfg['label']}: {out.shape}"
    print("\nAll forward passes OK.")
