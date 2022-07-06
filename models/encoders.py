import torchvision
from torch import nn


class EfficientNetEncoder(nn.Module):
    def __init__(self, variant: int):
        assert 0 <= variant <= 7
        super().__init__()
        self.model = getattr(
            torchvision.models,
            f"efficientnet_b{variant}",
        )(pretrained=True, stochastic_depth_prob=0)
        self.model.classifier = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
