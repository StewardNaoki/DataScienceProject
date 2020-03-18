import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss


def dice_loss(a, b, weight=1, log=False):
    a = a.view(a.shape[0], -1)
    b = b.view(a.shape[0], -1)
    # print(torch.sum(a*b))
    # print(torch.sum(a))
    # print(torch.sum(b))
    intersect = a*b
    score = (2 * torch.sum(weight * intersect) /
             (torch.sum(weight * a) + torch.sum(weight * b)))
    if log:
        return -torch.log(score)
    else:
        return 1 - score


class Custom_loss:
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        self.BCE_weight = 0.2
        self.dice_weight = 1
        # self.f_loss = nn.BCELoss()
        pos_weight = torch.tensor([self.BCE_weight])
        self.f_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()

    def __call__(self, inputs, targets):  # voir BCEWithLogitsLoss
        outputs = self.f_loss(inputs, targets) + dice_loss(self.sigmoid(inputs), targets, self.dice_weight )
        # outputs.backward(retain_graph=True)
        return outputs
