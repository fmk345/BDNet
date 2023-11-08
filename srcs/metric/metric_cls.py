# -----------------------------------------
# 🎯 Metrics for object classification tasks
# -----------------------------------------

import torch
import torch.nn as nn

# 💡 the inputs and outputs are in 'torch tensor' format


def accuracy(output, target):
    '''
    calculate classification accuracy
    '''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert len(pred) == len(target)
        correct = torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    '''
    calculate top-K classification accuracy
    '''
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert len(pred) == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
