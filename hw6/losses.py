import sys
import torch
from torch import nn


EPS = 1e-8


class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, y_pred, y_true):
        sum_true = torch.sum(y_true, dim=(1, 2, 3), keepdim=True)
        y_true = y_true / (EPS + sum_true)

        sum_pred = torch.sum(y_pred, dim=(1, 2, 3), keepdim=True)
        y_pred = y_pred / (EPS + sum_pred)

        loss = y_true * torch.log(EPS + y_true / (EPS + y_pred))
        loss = torch.mean(torch.sum(loss, dim=(1, 2, 3)))

        return loss


class CCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        a = (y_pred - torch.mean(y_pred, axis=(1, 2, 3), keepdims=True)) / (
            torch.std(y_pred, axis=(1, 2, 3), keepdims=True) + EPS
        )
        b = (y_true - torch.mean(y_true, axis=(1, 2, 3), keepdims=True)) / (
            torch.std(y_true, axis=(1, 2, 3), keepdims=True) + EPS
        )
        r = torch.sum(a * b, axis=(1, 2, 3), keepdims=True) / torch.sqrt(
            (a * a).sum(axis=(1, 2, 3), keepdims=True)
            * (b * b).sum(axis=(1, 2, 3), keepdims=True)
            + EPS
        )
        return -r.mean()
    
    
class SIMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SIMLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred / (torch.sum(y_pred, axis=(1, 2, 3), keepdims=True) + 1e-7)
        y_true = y_true / (torch.sum(y_true, axis=(1, 2, 3), keepdims=True) + 1e-7)
        r = torch.sum(torch.minimum(y_pred, y_true), axis=(1, 2, 3), keepdims=True)
        return -r.mean()
    
    
class NSSLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NSSLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        pred_map_ = (y_pred - torch.mean(y_pred, axis=(1, 2, 3), keepdims=True)) / torch.std(y_pred, axis=(1, 2, 3), keepdims=True)
        mask = y_true.gt(0)
        score = torch.mean(torch.masked_select(pred_map_, mask))
        return -score
    
class MultiLoss(nn.Module):
    def __init__(self, alpha_kld=1, alpha_nss=1, alpha_sim=1, alpha_cc=1):
        super(MultiLoss, self).__init__()
        self.alpha_kld = alpha_kld
        self.alpha_nss = alpha_nss
        self.alpha_sim = alpha_sim
        self.alpha_cc = alpha_cc
        
        self.kld_loss = KLDLoss()
        self.sim_loss = SIMLoss()
        self.nss_loss = NSSLoss()
        self.cc_loss = CCLoss()

    def forward(self, y_pred, y_true):
        kld_loss = self.kld_loss(y_pred, y_true)
        sim_loss = self.sim_loss(y_pred, y_true)
        nss_loss = self.nss_loss(y_pred, y_true)
        cc_loss = self.cc_loss(y_pred, y_true)
        
        total_loss = 0 \
            + self.alpha_kld * kld_loss \
            + self.alpha_nss * nss_loss \
            + self.alpha_sim * sim_loss \
            + self.alpha_cc * cc_loss \
                
        return dict(
            total=total_loss,
            kld=kld_loss,
            nss=nss_loss,
            cc=cc_loss,
            sim=sim_loss
        )
        