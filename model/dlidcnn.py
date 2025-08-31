import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DLID_CNN(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=10000):
        super(DLID_CNN, self).__init__()
        base = models.resnet18(pretrained=False)
        base.fc = nn.Linear(base.fc.in_features, embedding_dim)
        self.backbone = base
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = F.normalize(feat)  # L2 normalize embeddings
        logits = self.fc(feat)
        return feat, logits
