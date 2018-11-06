import numpy as np
import utils
from torch import nn
import torch


def validation_binary(model: nn.Module, criterion, valid_loader):
    with torch.no_grad():
        model.eval()
        losses = []

        iou = []

        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            iou += [iou_pytorch((outputs > 0.5), targets).item()]

        valid_loss = np.mean(losses)  # type: float

        mean_iou = np.mean(iou).astype(np.float64)

        print('Valid loss: {:.5f}, mean_iou: {:.5f}'.format(valid_loss,mean_iou))
        metrics = {'valid_loss': valid_loss, 'mean_iou': mean_iou}
        return metrics


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    epsilon = 1e-6
    labels = labels.byte()

    intersection = (outputs & labels).float().sum((-2, -1))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((-2, -1))  # Will be zero if both are 0

    iou = (intersection + epsilon) / (union + epsilon)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds

    return thresholded.mean()
