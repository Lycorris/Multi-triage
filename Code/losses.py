import torch
from torch import nn
"""
Losses:
    (pred_logits, y_binarized)
 - 1. BCEWithLogitsLoss()
 - 2. CrossEntropyLoss()
 - 3. MultiLabelSoftMarginLoss()
 - 4. BCEWithMSELoss()
 - 5. CustomizedBCE()
 - 6. AsymmetricLossOptimized()
"""


class BCEWithMSELoss(nn.Module):
    """
    weighted sum of nn.BCEWithLogitsLoss() and nn.MSELoss().

    a weight_MSE of *0.5 ~ 0.8* is recommended.
    """

    def __init__(self, weight_BCE=0.3, weight_MSE=0.7, sigmoid=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_BCE = weight_BCE
        self.weight_MSE = weight_MSE
        self.sigmoid = sigmoid

    def forward(self, x, y):
        """Parameters

        @param x: input logits
        @param y: targets (multi-label binarized vector)
        """
        loss_BCE = nn.BCEWithLogitsLoss()(x, y)
        if self.sigmoid:
            x = nn.Sigmoid()(x)
        loss_MSE = nn.MSELoss()(x, y)
        loss = self.weight_BCE * loss_BCE + self.weight_MSE * loss_MSE
        return loss


class CustomizedBCELoss(nn.Module):
    """
    a flexible version of BCE,
    which enable the loss to focus more on the performance of positive samples' prediction
    """

    def __init__(self, weight_pos=0.8, weight_neg=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, x, y):
        x = nn.Sigmoid()(x)
        loss_pos = y * torch.log(x)
        loss_neg = (1 - y) * torch.log(1 - x)
        loss = self.weight_pos * loss_pos + self.weight_neg * loss_neg
        return -torch.sum(loss)


class AsymmetricLossOptimized(nn.Module):
    """
    AsymmetricLoss from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py

    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


if __name__ == '__main__':
    # TODO: sanity test
    y = torch.tensor([0, 1, 0])
