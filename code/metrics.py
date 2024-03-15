import torch


def metrics_acc(y: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5):
    """

    Args:
        y: the actual y
        y_pred: the pred of y

    Returns:
        e.g. y = [0, 1, 0, 1, 0, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 0, 1]
        metrics_acc(y, y_pred) = 2 / 5 = 0.4

    """
    pred = torch.where(y_pred > threshold, 1, 0)
    correct = torch.sum(y * pred, dim=1)
    all = torch.sum(y, dim=1) + torch.sum(pred, dim=1) - correct
    return torch.mean(correct / all)

if __name__ == '__main__':
    y = torch.tensor([[1, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1]])
    y_pred = torch.tensor([[0.8, 0.2, 0.9, 0.4, 0.3, 0.7, 0.8], [0, 0, 0, 1, 1, 0, 1]])
    print(metrics_acc(y, y_pred))