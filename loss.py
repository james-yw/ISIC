import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss,self).__init__()


    def forward(self, probs, targets):

        num = targets.size(0)
        smooth = 1.

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)

        intersection = (m1*m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

if __name__=="__main__":
    pred = torch.randn(2, 1, 320, 320)
    target = torch.randn(2, 1, 320, 320)

    Diceloss = SoftDiceLoss()
    loss = Diceloss(pred,target)

    print("loss_shape:",loss)