import os

import cv2
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from torch import nn
from torchvision import transforms

from utils import padding


class SaliencyModel(nn.Module):
    def __init__(self):
        """Initializes model's architecture.
        Assumes an input to be (3, height, width) tensor
        """
        super(SaliencyModel, self).__init__()

        # Encoder
        # Берем предобученный Deeplabv3
        deeplab = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )

        self.backbone = deeplab.backbone
        self.aspp = deeplab.classifier[0]

        # Можем убрать слой с Dropout
        self.aspp.project[-1] = nn.Identity()

        # Decoder для получения одноканального изображения
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.dec4 = nn.Conv2d(32, 1, kernel_size=3, padding="same")


    def forward(self, x):
        x = self.backbone(x)["out"]
        x = self.aspp(x)
        # 1/8 от исходного разрешения
        x = F.interpolate(
            self.dec1(x), scale_factor=2, mode="bilinear", align_corners=True
        )
        # 1/4 от исходного разрешения
        x = F.interpolate(
            self.dec2(x), scale_factor=2, mode="bilinear", align_corners=True
        )
        # 1/2 от исходного разрешения
        x = F.interpolate(
            self.dec3(x), scale_factor=2, mode="bilinear", align_corners=True
        )
        # 1/1 от исходного разрешения
        x = self.dec4(x)
        # 0-1 нормализация
        return self.normalize(x)

    def normalize(self, x, eps=1e-8):
        shapes = x.shape
        x = x.view(shapes[0], -1)
        x = (x - x.min(dim=1, keepdim=True).values) / (
            x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values + eps
        )
        x = x.view(shapes)
        return x

class SaliencyEvaluator(object):
    def __init__(self, model_path="saliency.pth"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SaliencyModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device).eval().float()

        # Не забываем повторить преобразования во время инференса
        self.im_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def evaluate(self, video_frames_path, output_saliency_path):

        for frame_name in tqdm.tqdm(sorted(os.listdir(video_frames_path))):

            frame_path = os.path.join(video_frames_path, frame_name)

            # Размер входа при тестировании 360x640
            frame = (
                padding(
                    cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB),
                    height=360,
                    width=640,
                ).astype("float32")
                / 255.0
            )
            frame = self.im_transform(frame).to(self.device)

            # Предсказываем карты внимания
            pred_sal = self.model(frame[None])[0, 0].cpu().numpy() * 255

            # Сохраняем результат
            cv2.imwrite(os.path.join(output_saliency_path, frame_name), pred_sal)
