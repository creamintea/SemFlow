import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(x + self.block(x))
    
class CTPInputAdapter(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, hidden_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        return self.net(x)
    
class CTPOutputAdapter(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 15, 3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        return self.net(x)