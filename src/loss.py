import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss


def dice_loss(a,b):
    a = a.view(a.shape[0],-1)
    b = b.view(a.shape[0],-1)
    # print(torch.sum(a*b))
    # print(torch.sum(a))
    # print(torch.sum(b))
    score = (2 * torch.sum(a*b) / (torch.sum(a) + torch.sum(b)))
    return 1 - score



class Custom_loss:
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        # self.f_loss = nn.BCELoss()
        self.f_loss = nn.BCEWithLogitsLoss()

    def __call__(self, inputs, targets):  # voir BCEWithLogitsLoss
        
        outputs = self.f_loss(inputs, targets) + dice_loss(inputs, targets)
        # outputs.backward(retain_graph=True)
        return outputs