import torch
import torchvision.transforms.functional as TF
from collections import defaultdict


import sys
import torch
import torch.nn as nn

from torch import nn

class SimpleRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
            
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x):
        return self.net(x)
    
    def loss(self, ref, dist):
        restored = self.forward(dist)
        mse = nn.functional.mse_loss(restored, ref)
        return {
            'mse': mse
        }

    def num_parameters(self) -> int:
        return sum(x.numel() for x in self.parameters())
    
    def train_epoch(
        self,
        train_loader: object,
        optimizer: object,
        use_cuda: bool,
        loss_key: str = "total",
    ) -> defaultdict:
        self.train()

        stats = defaultdict(list)
        for batch in train_loader:
            x, y = batch['attacked_image'], batch['reference_image']
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            losses = self.loss(y, x)
            optimizer.zero_grad()
            losses[loss_key].backward()
            optimizer.step()

            for k, v in losses.items():
                stats[k].append(v.item())

        return stats

    def eval_model(self, data_loader: object, use_cuda: bool) -> defaultdict:
        self.eval()
        stats = defaultdict(float)
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch['attacked_image'], batch['reference_image']
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                losses = self.loss(y, x)
                for k, v in losses.items():
                    stats[k] += v.item() * x.shape[0]

            for k in stats.keys():
                stats[k] /= len(data_loader.dataset)
        return stats

    
class Defense:
    def __init__(self, device: str) -> None:
        """
            device: str = "cuda" or "cpu", передается на случай использования нейронных сетей
        """
        self.device = device
        
    def forward(self, images):
        with torch.no_grad():
            alpha = 1 / 2
            image_refined_1 = model(images).clamp(0.0, 1.0)
            image_mixed_1 = images * (1 - alpha) + image_refined_1 * alpha
            
            # image_refined_2 = model(image_mixed_1)
            # image_mixed_2 = images * (1 - alpha) + image_refined_2 * alpha
        return image_mixed_1

    def apply_defense(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images).to(self.device)
    
model = SimpleRefiner()
model.load_state_dict(torch.load('simple_model_v2.pth'))
model.eval()
model.cuda()