import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        if isinstance(weight, list):
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, input, target):
        return self.loss(input, target)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None):
        super(BCEWithLogitsLoss, self).__init__()
        if isinstance(weight, list) or isinstance(weight, np.ndarray):
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, input, target):
        return self.loss(input, target)


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, question_weight=0.5, answer_weight=0.5):
        super(CustomBCEWithLogitsLoss, self).__init__()
        if isinstance(weight, list):
            weight = torch.tensor(weight, dtype=torch.float)
        self.criterion = nn.BCEWithLogitsLoss(weight, size_average)
        self.question_weight = question_weight
        self.answer_weight = answer_weight
        print(self.question_weight, self.answer_weight)
        assert self.question_weight + self.answer_weight == 1

    def forward(self, input, target):
        loss1 = self.criterion(input[:, 0:9], target[:, 0:9])
        loss2 = self.criterion(input[:, 9:10], target[:, 9:10])
        loss3 = self.criterion(input[:, 10:21], target[:, 10:21])
        loss4 = self.criterion(input[:, 21:26], target[:, 21:26])
        loss5 = self.criterion(input[:, 26:30], target[:, 26:30])
        loss = self.question_weight * (loss1 + loss3 + loss5) + self.answer_weight * (loss2 + loss4)
        return loss


class OrdinalLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
        self.loss = F.binary_cross_entropy_with_logits

    def forward(self, input, target):
        ordinal_target = torch.zeros((target.size(0), self.num_classes))
        for i, label in enumerate(target):
            ordinal_target[i] = torch.cat([torch.ones(label), torch.zeros(self.num_classes - label)])
        ordinal_target = ordinal_target.type(torch.cuda.HalfTensor)
        return self.loss(input, ordinal_target)


class MSELoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(size_average, reduce, reduction)

    def forward(self, input, target):
        target = target.type(torch.cuda.FloatTensor)
        return self.loss(input, target)


class WeightedMSELoss(nn.Module):
    def __init__(self, loss_weights):
        super(WeightedMSELoss, self).__init__()
        if isinstance(loss_weights, list):
            self.weights = torch.tensor(loss_weights, dtype=torch.float)
        else:
            self.weights = loss_weights
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        batch_weights = torch.zeros(len(input)).to(torch.device('cuda'))
        for label in range(len(self.weights)):
             batch_weights[target == label] = self.weights[label]
        target = target.type(torch.cuda.FloatTensor).view(-1, 1)
        loss = torch.mean(batch_weights * (input - target) ** 2)
        return loss


class PenalizedMSELoss(nn.Module):
    def __init__(self, label_to_penalize, penalize_weight):
        super(PenalizedMSELoss, self).__init__()
        self.lower_bound = torch.tensor(label_to_penalize - 0.5, dtype=torch.float32).to(torch.device('cuda'))
        self.upper_bound = torch.tensor(label_to_penalize + 0.5, dtype=torch.float32).to(torch.device('cuda'))
        self.label_to_penalize = torch.tensor(label_to_penalize, dtype=torch.long).to(torch.device('cuda'))
        self.penalize_weight = torch.tensor(penalize_weight, dtype=torch.float32).to(torch.device('cuda'))
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        batch_weights = torch.zeros(len(input)).to(torch.device('cuda'))
        for i in range(len(input)):
            if self.lower_bound < input[i] < self.upper_bound and target[i] != self.label_to_penalize:
                batch_weights[i] = self.penalize_weight
            else:
                batch_weights[i] = 1

        target = target.type(torch.cuda.FloatTensor).view(-1, 1)
        loss = torch.mean(batch_weights * (input - target) ** 2)
        return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, loss_weights):
        super(MultiTaskLoss, self).__init__()
        self.loss_weights = loss_weights
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        loss_mse = self.mse_loss(input[:, 0], target.to(torch.float))
        loss_ce = self.cross_entropy_loss(input[:, 1:], target.to(torch.long))
        loss = (loss_mse * self.loss_weights[0] + loss_ce * self.loss_weights[1]) / sum(self.loss_weights)
        return loss
