import torch
import torch.nn as nn


class TinyAlexNet(nn.Module):
    """
    Custom AlexNet implementation

    # Parameters
        num_classes: int to properly setup the last layer
        keep_droupout_prob: float to set the probability of keeping a neuron
                                    in the config of the network during each
                                    forward pass
    """

    def __init__(self, num_classes: int = 1, drop_prob: float = 0.5):
        super(TinyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(32 * 6 * 6, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(16, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
