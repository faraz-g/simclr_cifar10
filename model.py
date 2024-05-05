import torch.nn as nn
from torchvision.models import resnet18


class SimCLR(nn.Module):
    def __init__(self, projection_dim: int = 128):
        super(SimCLR, self).__init__()

        self.feature_extractor = resnet18(pretrained=False)
        self.n_features = self.feature_extractor.fc.in_features

        self.feature_extractor.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return z_i, z_j
