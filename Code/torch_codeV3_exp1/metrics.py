import torch
from torch import nn
'''
    metrics: acc, P, R, F1
'''
def top_k_accuracy(y_true, y_pred, topk=(1,)):
    maxk = max(topk)
    if y_true.shape[1] < maxk:
        maxk = y_true.shape[1]
        print(f"maxk is set to {maxk} because y_true.shape[1] < maxk")

    _, pred = y_pred.topk(maxk, dim=1, largest=True, sorted=True)
    ret = []
    for k in topk:
        correct = (y_true * torch.zeros_like(y_true).scatter(1, pred[:, :k], 1)).float()
        ret.append(correct.sum() / y_true.sum())
    return ret

def metrics(y: torch.Tensor, pred: torch.Tensor, split_pos: list, threshold: float = 0.5, from_logits=True):
    if from_logits:
        pred = nn.Sigmoid()(pred)
    original_pred = pred
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

    acc_at_1_d,acc_at_2_d,acc_at_3_d, acc_at_5_d, acc_at_10_d, acc_at_20_d = top_k_accuracy(y_d, pred_d, topk=(1, 2, 3, 5, 10, 20))
    acc_at_1_b,acc_at_2_b,acc_at_3_b, acc_at_5_b, acc_at_10_b, acc_at_20_b = top_k_accuracy(y_b, pred_b, topk=(1, 2, 3, 5, 10, 20))

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'F1': F1,
        'acc@1_d': acc_at_1_d.item(),
        'acc@2_d': acc_at_2_d.item(),
        'acc@3_d': acc_at_3_d.item(),
        'acc@5_d': acc_at_5_d.item(),
        'acc@10_d': acc_at_10_d.item(),
        'acc@20_d': acc_at_20_d.item(),
        'acc@1_b': acc_at_1_b.item(),
        'acc@2_b': acc_at_2_b.item(),
        'acc@3_b': acc_at_3_b.item(),
        'acc@5_b': acc_at_5_b.item(),
        'acc@10_b': acc_at_10_b.item(),
        'acc@20_b': acc_at_20_b.item(),
    }

# test top k accuracy
if __name__ == '__main__':
    y_true = torch.tensor([[1, 0, 0 ,1 ,1], [0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
    y_pred = torch.tensor([[0.9,0.5,0.2,0.1,0.2],[0.1,0.8,0.3,0.4,0.3],[0.1,0.2,0.9,0.1,0.1],[0.1,0.5,0.9,0.2,0.2]])
    # top 1: 4 / 11
    # top 2: 6 / 11
    # top 3: 7 / 11
    # top 5: 11/ 11
    print(top_k_accuracy(y_pred = y_pred, y_true = y_true, topk=(1,2,3,5)))