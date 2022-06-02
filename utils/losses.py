import torch
import torch.nn as nn
import math
from config import *

DEVICE=torch.device('cuda:'+str(cuda_id) if torch.cuda.is_available() else 'cpu')

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class ArcFaceLoss(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits

class ElasticArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50,std=0.0125, is_cuda=False):
        super(ElasticArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.std=std
        self.criterion = torch.nn.CrossEntropyLoss()
        if is_cuda:
            self.criterion = self.criterion.to(DEVICE)

    def forward(self, input, label):
        cos_theta = input.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return self.criterion(cos_theta, label) 

class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

def get_loss(name: str = 'ArcFace'):
    if name == 'ArcFace':
        loss_function = ArcFaceLoss()
    elif name == 'ElasticArcFace':
        loss_function = ElasticArcFaceLoss()
    elif name == 'FocalLoss':
        loss_function = FocalLoss()
    elif name == 'CosFace':
        loss_function = CosFace()
    else:
        loss_function = nn.CrossEntropyLoss()
    return loss_function