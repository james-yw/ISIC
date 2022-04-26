import torch
import numpy as np

def accuracy(pred, target):
    pred, target = pred.numpy(), target.numpy()
    pred = np.argmax(pred, axis=1)
    acc = pred.eq(target)
