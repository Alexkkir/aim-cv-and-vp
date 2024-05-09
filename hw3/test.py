import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import auc
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# PAQ-2-PIQ metric
from paq2piq_standalone import MetricModel
PAQ2PIQ_METRIC_RANGE = 150

EPS = 1e-6

transform_to_tensor = transforms.Compose([
    transforms.ToTensor()
    ])
# Import attacks
from iterative import attack as iterative_attack
from uap import read_uap_patch
from uap import attack as uap_attack


# Util functions
def to_torch(x, device='cpu'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)
        x = x.type(torch.FloatTensor).to(device)
    return x


def to_numpy(x):
    if torch.is_tensor(x):
        x = x.cpu().detach().permute(0, 2, 3, 1).numpy()
    return x if len(x.shape) == 4 else x[np.newaxis]


# Returns image compressed with JPEG with quality factor q (np.array if return_torch=False, torch.Tensor otherwise)
def compress(img, q, return_torch=False):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    np_batch = to_numpy(img)
    if len(np_batch.shape) == 3:
        np_batch = np_batch[np.newaxis]
    jpeg_batch = np.empty(np_batch.shape)
    for i in range(len(np_batch)):
        result, encimg = cv2.imencode('.jpg', np_batch[i] * 255, encode_param)
        jpeg_batch[i] = cv2.imdecode(encimg, 1) / 255
    return torch.nan_to_num(to_torch(jpeg_batch), nan=0) if return_torch else np.nan_to_num(jpeg_batch, nan=0)


# Main test function
def test_attack(attack_callback, model, dataset_path='./public_dataset',
                qfs=np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                epsilons=np.array([2, 4, 8, 10]) / 255.0, device='cuda:0', attack_type='iterative',
                metric_range=100,
                uap_train_path='./pretrained_uap_paq2piq.png',
                csv_results_dir=None):
    """
    Main test function.\n
    Args:\n
    attack_callback (callable): Function that takes clear image (torch.Tensor of shape [1,3,H,W]) and returns corresponding adversarial example (same type and shape):\n
    model: (PyTorch model): Metric model to be attacked. Should be an object of a class that inherits torch.nn.module and has a forward method that supports backpropagation.\n
    dataset_path: (str) Path to directory with images to test attack on\n
    qfs: (np.ndarray of floats/ints) quality factors to use in JPEG compression for rel.gain/qf curve estimation. Should contain values in range [0,100] in ascending order.
    epsilons: (np.ndarray of floats/ints) List of epsilons (maximum allowed pixel-wise diffence between clear and attacked images) to use during testing.\n
    device (str or torch.device()): Device to use in computaions.\n
    attack_type: ('iterative' or 'uap') Whether to use iterative attack call (metric model is available in attack_callback) or UAP (additionaly loads pretrained adversarial patch)\n
    uap_train_path: (str) path to file with pretrained universal adversarial additive (.png or something else, read function is user-defined in ```uap.py``` (read_uap_patch())).\n
    csv_results_dir: (str) path to save directory for csvs with raw results. If None, results are not saved.\n
    Returns:\n
        float: total score for given attack on given test set.
    """
    assert attack_type in ['iterative', 'uap']

    # Read pretrained UAP patch
    if attack_type == 'uap':
        uap_patch = read_uap_patch(uap_train_path)
    
    model.to(device)
    model.eval()
    results = pd.DataFrame(columns=['image_name', 'eps', 'score'])

    # Collect mean pixel-wise image differences between clear and attacked for statistics
    diffs = {int(x * 255):[] for x in epsilons}

    # Iterate over image dataset
    for image_path in tqdm(Path(dataset_path).iterdir(), total=len([x for x in Path(dataset_path).iterdir()])):
        if not Path(image_path).suffix in ('.png', '.jpg', '.jpeg'):
            continue

        im = Image.open(image_path)
        im_tensor = transform_to_tensor(im).unsqueeze(0).to(device)
        with torch.no_grad():
            #clear_val = model(im_tensor.to(device)).item()
            # Clear compressed metric values
            compressed_clear_vals = np.array([model(compress(im_tensor, q, return_torch=True).to(device)).item() for q in qfs])
        
        scores_sum = 0
        for eps in epsilons:
            if attack_type == 'iterative':
                im_attacked = attack_callback(im_tensor, model, metric_range=metric_range, device=device, eps=eps)
            else:
                im_attacked = attack_callback(image=im_tensor, uap_patch=uap_patch, device=device, eps=eps)
            with torch.no_grad():
                # Enforce [-eps, eps] limit on differences
                diff = im_attacked - im_tensor
                diff = torch.clamp(diff, min=-eps, max=eps)
                im_attacked = im_tensor + diff

                attacked_vals = np.array([model(compress(im_attacked, q, return_torch=True).to(device)).item() for q in qfs])
                rel_gains = (attacked_vals - compressed_clear_vals) / (compressed_clear_vals + EPS)
                # AUC score for current image-eps pair
                score = auc(qfs, rel_gains)
                scores_sum += score

                # add row to results df
                results.loc[len(results.index)] = [Path(image_path).stem, eps, score] 
                # Save mean differences
                diffs[int(eps * 255)].append(float(torch.abs(diff).mean().detach().cpu().item()))
        print(scores_sum / len(epsilons))

    # Print mean image diffs for different epsilons
    for int_eps in diffs.keys():
        print(f'eps={int_eps}/255 :', 'mean diff = {:.5f}'.format(np.array(diffs[int_eps]).mean()), 'mean ')
    
    # Save raw results to csv *attack_type*.csv in csv_results_dir directory.
    if csv_results_dir is not None:
        csv_path = os.path.join(csv_results_dir, 'results.csv')
        results.to_csv(csv_path)
        print(f'Results saved to {csv_path}')

    # Mean for all images over all epsilons
    #mean_results = results.groupby(['image_name', 'eps']).agg('mean').reset_index()
    mean_results = results.groupby(['image_name']).agg('mean').reset_index()
    # print(mean_results)

    # return mean AUC score over all images
    return mean_results['score'].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", type=str, default="iterative")
    parser.add_argument("--uap_train_path", type=str, default="../uap_trained_data/pretrained_uap_paq2piq.png")
    parser.add_argument("--csv_results_dir", type=str, default=None)
    parser.add_argument("--model_weights", type=str, default='../weights/RoIPoolModel.pth')
    parser.add_argument("--dataset_path", type=str, default='../NIPS_test/')
    parser.add_argument("--device", type=str, default='cuda:0')

    args = parser.parse_args()
    device = args.device

    model = MetricModel(device, model_path=args.model_weights).to(device)
    metric_range = PAQ2PIQ_METRIC_RANGE
    model.eval()
    if args.attack_type == 'iterative':
        attack_func = iterative_attack
    elif args.attack_type == 'uap':
        attack_func = uap_attack
    else:
        raise ValueError('Wrong attack type. Only "iterative" and "uap" are supported.')

    total_score = test_attack(attack_func, model=model, dataset_path=args.dataset_path,
                            device=device, attack_type=args.attack_type, metric_range=metric_range,
                            uap_train_path=args.uap_train_path, csv_results_dir=args.csv_results_dir)
    print('Result for {} type attack: {:.4f}'.format(args.attack_type.capitalize(), total_score))
    