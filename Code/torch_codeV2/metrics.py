import torch
from torch import nn
'''
    metrics: acc, P, R, F1
'''

def metrics(y: torch.Tensor, pred: torch.Tensor, split_pos: list, threshold: float = 0.5, from_logits=True):
    if from_logits:
        pred = nn.Sigmoid()(pred)
    pred = torch.where(pred > threshold, 1, 0)

    y_d, y_b = torch.split(y, split_pos, dim=1)
    pred_d, pred_b = torch.split(pred, split_pos, dim=1)

    TPd, TPb = torch.sum(y_d * pred_d, dim=1), torch.sum(y_b * pred_b, dim=1)
    TNd, TNb = torch.sum((1 - y_d) * (1 - pred_d), dim=1), torch.sum((1 - y_b) * (1 - pred_b), dim=1)
    FPd, FPb = torch.sum((1 - y_d) * pred_d, dim=1), torch.sum((1 - y_b) * pred_b, dim=1)
    FNd, FNb = torch.sum(y_d * (1 - pred_d), dim=1), torch.sum(y_b * (1 - pred_b), dim=1)

    acc = torch.mean((TPd + TNd) / (TPd + TNd + FPd + FNd + 1e-6)).item(), torch.mean(
        (TPb + TNb) / (TPb + TNb + FPb + FNb + 1e-6)).item()
    recall = torch.mean(TPd / (TPd + FNd + 1e-6)).item(), torch.mean(TPb / (TPb + FNb + 1e-6)).item()
    precision = torch.mean(TPd / (TPd + FPd + 1e-6)).item(), torch.mean(TPb / (TPb + FPb + 1e-6)).item()
    F1 = 2 * recall[0] * precision[0] / (recall[0] + precision[0] + 1e-6), 2 * recall[1] * precision[1] / (
            recall[1] + precision[1] + 1e-6)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'F1': F1
    }