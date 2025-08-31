import torch
import torch.nn as nn
import torch.nn.functional as F

class DLIDLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=512, margin=0.5, lambda_intra=1.0, lambda_inter=0.5):
        super(DLIDLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.margin = margin
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]

        # Intra-class loss
        intra_loss = F.mse_loss(features, centers_batch)

        # Inter-class loss
        dist_matrix = torch.cdist(features, self.centers)  # (batch, num_classes)
        mask = torch.ones_like(dist_matrix, device=features.device)
        mask[torch.arange(batch_size), labels] = 0
        inter_loss = F.relu(self.margin - dist_matrix * mask).mean()

        loss = self.lambda_intra * intra_loss + self.lambda_inter * inter_loss
        return loss
