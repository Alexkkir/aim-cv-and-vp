import os
import cv2
import tqdm
import numpy as np
from torch import nn
from collections import defaultdict

# Приводит изображение в диапазон 0-1
def normalize_map(s_map):
    norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0 + 1e-7)
    return norm_s_map


# Паддинг черными краями до нужного размера, не меняет пропорции
def padding(img, height=216, width=384, channels=3):
    channels = img.shape[2] if len(img.shape) > 2 else 1

    if channels == 1:
        img_padded = np.zeros((height, width), dtype=img.dtype)
    else:
        img_padded = np.zeros((height, width, channels), dtype=img.dtype)

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = cv2.resize(img, (new_cols, height))
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):
                   ((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        img = cv2.resize(img, (width, new_rows))
        if new_rows > height:
            new_rows = height
        img_padded[((img_padded.shape[0] - new_rows) // 2):
                   ((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=216, cols=384):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        out[min(r, rows-1), min(c, cols-1)] = 255

    return out.astype(int)


# Паддинг карт фиксаций
def padding_fixation(img, height=216, width=384):
    img_padded = np.zeros((height, width))

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = resize_fixation(img, rows=height, cols=new_cols)
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):
                   ((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=width)
        if new_rows > height:
            new_rows = height
        img_padded[((img_padded.shape[0] - new_rows) // 2):
                   ((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def nss(s_map, gt):
    x,y = np.where(gt)
    s_map_norm = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    temp = []
    for i in zip(x,y):
        temp.append(s_map_norm[i[0], i[1]])
    return np.mean(temp)

    
def similarity(s_map, gt):
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return np.sum(np.minimum(s_map, gt))

    
def cc(s_map, gt):
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r    


# Функция вычисления тестовых метрик
def calculate_single_observer_metrics(test_output_path, gt_path, num_observers=3):
    """Calculates average CC, SIM, NSS on the passed data

    Parameters
    ----------
        test_output_path : string
            path to predicted saliency
        gt_path : string
            path to ground-truth saliency & fixations  
    Returns
    -------
        (CC, SIM, NSS metrics dict)
    """
    global similarity, nss, cc

    assert sorted(os.listdir(test_output_path)) == sorted(os.listdir(gt_path))

    # Per-video metrics
    sim_nss_cc = []

    for video_name in sorted(tqdm.tqdm(os.listdir(gt_path))):

        observers_pred_paths = os.listdir(os.path.join(test_output_path, video_name))

        assert len(observers_pred_paths) == num_observers


        gt_saliency_path = os.path.join(gt_path, video_name, 'gt_saliency')
        gt_fixations_path = os.path.join(gt_path, video_name, 'gt_fixations')

        # Per-observer metrics
        sim_nss_cc_ = []

        for observer_id in sorted(observers_pred_paths):
            
            pred_video_path = os.path.join(test_output_path, video_name, observer_id)

            # Проверям, что число и имена кадров одинаковые
            assert set([x for x in os.listdir(pred_video_path) if '.png' in x]) ==\
                   set([x for x in os.listdir(gt_saliency_path) if '.png' in x]) ==\
                   set([x for x in os.listdir(gt_fixations_path) if '.png' in x])
            
            # Per-frame metrics
            sim_nss_cc__ = []

            for frame in tqdm.tqdm(sorted(os.listdir(gt_saliency_path))):

                gt_fix = cv2.imread(os.path.join(gt_fixations_path, frame), 0)
                gt_sm = cv2.imread(os.path.join(gt_saliency_path, frame), 0)
                pred_sm = cv2.imread(os.path.join(pred_video_path, frame), 0)
                
                gt_fix = padding_fixation(gt_fix, 360, 640)
                gt_sm = normalize_map(padding(gt_sm, 360, 640))
                pred_sm = normalize_map(padding(pred_sm, 360, 640))

                sim_val = similarity(pred_sm, gt_sm)
                nss_val = nss(pred_sm, gt_fix)
                cc_val = cc(pred_sm, gt_sm)
                
                sim_nss_cc__.append([sim_val, nss_val, cc_val])

            sim_nss_cc_.append(np.array(sim_nss_cc__).mean(axis=0))

        sim_nss_cc.append(np.array(sim_nss_cc_).mean(axis=0))

    sim, nss, cc = np.array(sim_nss_cc).mean(axis=0)

    return  {'sim': sim, 'nss': nss, 'cc': cc}


def detach_dict(x):
    return {k: v.item() for k, v in x.items()}

def add_prefix(x, s):
    return {s + k: v for k, v in x.items()}

def reduce_dict(lst_dict):
    out = defaultdict(float)
    for d in lst_dict:
        for k, v in d.items():
            out[k] += v
            
    for k in out:
        out[k] /= len(lst_dict)
    return out