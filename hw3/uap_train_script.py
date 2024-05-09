from uap import train
import argparse
# PAQ-2-PIQ metric
from paq2piq_standalone import MetricModel
PAQ2PIQ_METRIC_RANGE = 150

import cv2
import os
def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_train", type=str, default="training_dataset")
    parser.add_argument("--save_path", type=str, default="uap_trained_data/pretrained_uap_paq2piq.png")
    parser.add_argument("--model_weights", type=str, default='weights/RoIPoolModel.pth')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda:0')

    args = parser.parse_args()
    device = args.device

    model = MetricModel(device, model_path=args.model_weights).to(device)
    metric_range = PAQ2PIQ_METRIC_RANGE
    model.eval()

    uap_trained_patch = train(model, args.path_train, batch_size=args.batch_size, metric_range=metric_range, device=args.device)
    #save_path = os.path.join(args.save_dir, "pretrained_uap_paq2piq.png")
    print(f'Saving pretrained patch to {args.save_path}')
    cv2.imwrite(args.save_path, (uap_trained_patch + 0.5) * 255)

if __name__ == "__main__":
    train_main()
