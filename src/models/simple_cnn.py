import torch
import torch.nn as nn
import torch
class simpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Sequential(self.make_block(in_channels=3, out_channels=8),
                     self.make_block(in_channels=8, out_channels=16),
                     self.make_block(in_channels=16, out_channels=32),
                     self.make_block(in_channels=32, out_channels=64),
                     self.make_block(in_channels=64, out_channels=128))
        self.Flatten = nn.Flatten()
        self.fc1  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=6272, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=10))

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same", stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same", stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2))
    def forward(self , x):
        x = self.conv1(x)
        x= self.Flatten(x)
        x = self.fc1(x)
        return x


