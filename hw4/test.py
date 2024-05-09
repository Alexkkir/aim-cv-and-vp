import argparse
import json
import os
import sys
import warnings
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import MetricModel
from defense import Defense

warnings.filterwarnings("ignore")

class SimpleLogger:
    def __init__(self, level: str ="full"):
        self.metrics = {}
        self.level = level

    def log(self, data: Dict[str, List[float]]):
        for key, values in data.items():
            if key not in self.metrics:
                self.metrics[key] = values
            else:
                self.metrics[key].extend(values)
                
    def save_scores(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.metrics)

        all_mean = df.mean(numeric_only=True).values.tolist()
        corr_per_attack = df.groupby('Attack Name').apply(
            lambda x: x[['Source Linearity', 'Defense Linearity']].corr(method='spearman').iloc[0, 1]
        )

        df = df.groupby('Attack Name').mean()
        df["SRCC"] = corr_per_attack
        df.loc["All Attacks"] = all_mean + [np.mean(corr_per_attack)]
        print(df)
        if self.level == "full":
            df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)
        elif self.level == "simple":
            data = {
                k:v for k, v in zip(df.columns, all_mean + [np.mean(corr_per_attack)])
            }
            with open(os.path.join(output_dir, "scores.json"), "w") as f:
                json.dump(data, f)


class AttackedDataset(Dataset):
    def __init__(self, data_dir, reference_dir, transform=None):
        """
        Arguments:
            data_dir (string): Directory with all the images.
            reference_dir (string): Directory with reference (clear) images.
            transform (bool): True for transformation to pass to model. False to return pristine image to display
        """
        print(data_dir)
        print(reference_dir)
        print(transform)
        
        self.data_dir = data_dir
        self.reference_dir = reference_dir
        self.transform = transform

    def __len__(self):
        return len(glob(os.path.join(self.data_dir, "*", "*")))

    def __getitem__(self, idx):
        img_name = sorted(glob(os.path.join(self.data_dir, "*", "*")))[idx]
        attacked_image = Image.open(img_name).convert("RGB")
        attack_name = img_name.split("/")[-2]

        if self.transform is not None:
            attacked_image = self.transform(attacked_image).type(torch.FloatTensor)

        reference_path = os.path.join(self.reference_dir, os.path.basename(img_name))
        reference_image = None
        if os.path.isfile(reference_path):
            reference_image = Image.open(reference_path).convert("RGB")

            if self.transform is not None:
                reference_image = self.transform(reference_image).type(torch.FloatTensor)
        else: 
            assert 0==1


        return {
            "attacked_image": attacked_image,
            "reference_image": reference_image,
            "attack_name": attack_name,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path attacked and reference images")
    parser.add_argument("--output_dir", type=str, help="path to reference images")
    parser.add_argument("--device", type=str, help="device to run metric model on", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=8, help="DataLoader's batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader's number of workers")
    parser.add_argument("--checkpoint_dir", type=str, default="./", help="Path to folder with Linearity checkpoint")
    parser.add_argument("--score_detail_level", type=str, default="full", help="Detail of the results output: save full scores.csv Dataframe with metrics for each attack or simple scores.json dict of averaged metrics", choices=["full", "simple"])
    args = parser.parse_args()

    defense = Defense(device=args.device)
    logger = SimpleLogger(level=args.score_detail_level)
    
    model = MetricModel(args.device, os.path.join(args.checkpoint_dir, "p1q2.pth"))
    model.eval();

    dataset = AttackedDataset(
        data_dir=os.path.join(args.data_dir, "attacked"),
        reference_dir=os.path.join(args.data_dir, "reference"),
        transform=transforms.ToTensor()
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    with torch.no_grad():
        for samples in tqdm(dataloader):

            attacked_images = samples["attacked_image"].to(args.device)
            source_images = samples["reference_image"].to(args.device)
            defended_images = defense.apply_defense(attacked_images.clone())

            defense_model_scores = model(defended_images).cpu().detach().squeeze(-1).numpy()
            attacked_model_scores = model(attacked_images).cpu().detach().squeeze(-1).numpy()
            source_image_scores = model(source_images).cpu().detach().squeeze(-1).numpy()

            if defended_images.shape != source_images.shape:
                defended_images = transforms.Resize(list(source_images.shape[2:]))(defended_images)

            source_images = source_images.cpu().detach().permute(0, 2, 3, 1).numpy()
            defended_images = defended_images.cpu().detach().permute(0, 2, 3, 1).numpy()

            psnr = [
                peak_signal_noise_ratio(source_image, defended_image, data_range=1)
                for source_image, defended_image in zip(source_images, defended_images)
            ]
            ssim = [
                structural_similarity(source_image, defended_image, channel_axis=2, data_range=1)
                for source_image, defended_image in zip(source_images, defended_images)
            ]

            quality_scores = np.array(ssim) + np.array(psnr) / 80.0
            gain_scores = np.abs(source_image_scores - defense_model_scores) / source_image_scores * 100.0
            logger.log(
                {
                    "Quality Score": (quality_scores / 2.).tolist(),
                    "Gain Score": gain_scores.tolist(),
                    "Source Linearity": source_image_scores.tolist(),
                    "Defense Linearity": defense_model_scores.tolist(),
                    "Attacked Linearity": attacked_model_scores.tolist(),
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "Attack Name": samples["attack_name"]
                }
            )

    logger.save_scores(args.output_dir)