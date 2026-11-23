import numpy as np
import torch


def evaluate(model, loader, device):
    """返回 (labels, probs)，顺序与 manifest 一致。"""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.softmax(logits.float(), dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)
