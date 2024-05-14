import torch
from torch import nn

'''
    metrics: acc, P, R, F1, acc@K(1, 2, 3, 5, 10, 20)
'''


def top_K_accuracy(y_true, y_pred, topK):
    maxK = max(topK)
    if y_true.shape[1] < maxK:
        maxK = y_true.shape[1]
        print(f"maxK is set to {maxK} because y_true.shape[1] < maxK")

    _, pred = y_pred.topk(maxK, dim=1, largest=True, sorted=True)
    ret = []
    for k in topK:
        correct = (y_true * torch.zeros_like(y_true).scatter(1, pred[:, :k], 1)).float()
        ret.append(correct.sum() / y_true.sum())
    return ret


def confusion_matrix(y: torch.Tensor, pred: torch.Tensor, threshold: float = 0.5, from_logits=True):
    if from_logits:
        pred = nn.Sigmoid()(pred)
    pred = torch.where(pred > threshold, 1, 0)

    eps = 1e-6
    TP = torch.sum(y * pred, dim=1)
    TN = torch.sum((1 - y) * (1 - pred), dim=1)
    FP = torch.sum((1 - y) * pred, dim=1)
    FN = torch.sum(y * (1 - pred), dim=1)

    acc = torch.mean((TP + TN) / (TP + TN + FP + FN + eps)).item()
    recall = torch.mean(TP / (TP + FN + eps)).item()
    precision = torch.mean(TP / (TP + FP + eps)).item()
    F1 = 2 * recall * precision / (recall + precision + eps)

    return acc, precision, recall, F1


def metrics(y, pred, split_pos, threshold=0.5, from_logits=True, topK=(1, 2, 3, 5, 10, 20)):
    """
        TODO:
    """
    res = []
    metric_name = []

    # split into Dev & Bug type
    y_d, y_b = torch.split(y, split_pos, dim=1)
    pred_d, pred_b = torch.split(pred, split_pos, dim=1)

    # a(acc), p(precision), r(recall), F(F1)
    metric_name.extend(['acc', 'precision', 'recall', 'F1'])
    a_p_r_F = list(zip(confusion_matrix(y_d, pred_d, threshold, from_logits),
                       confusion_matrix(y_b, pred_b, threshold, from_logits)))
    res.extend(a_p_r_F)

    # acc@(1, 2, 3, 5, 10, 20)
    metric_name.extend([f'acc@{k}' for k in topK])
    acc_at_n = list(zip(top_K_accuracy(y_d, pred_d, topK=topK),
                        top_K_accuracy(y_b, pred_b, topK=topK)))
    res.extend(acc_at_n)

    # map dict
    res_dict = {metric_name[i]: res[i] for i in range(len(res))}

    return res_dict


# test top k accuracy
if __name__ == '__main__':
    y_true = torch.tensor([[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
    y_pred = torch.tensor(
        [[0.9, 0.5, 0.1, 0.1, 0.2], [0.1, 0.8, 0.3, 0.4, 0.3], [0.1, 0.2, 0.9, 0.1, 0.1], [0.1, 0.5, 0.9, 0.2, 0.2]])
    # top 1: 4 / 11
    # top 2: 6 / 11
    # top 3: 7 / 11
    # top 5: 11/ 11
    # print(top_K_accuracy(y_pred=y_pred, y_true=y_true, topK=(1, 2, 3, 5)))
    print(metrics(y=y_true, pred=y_pred, split_pos=3, threshold=0.5, from_logits=False))
