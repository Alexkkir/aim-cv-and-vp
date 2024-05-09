import os
import cv2
import tqdm
import torch
import imageio
import itertools
import torchvision
import numpy as np
from torch import nn
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import normalize_map, padding, padding_fixation

class DatasetClass(Dataset):
    def __init__(self, path_data, transforms=None, num_frames=1, mode='val'):
        self.input_path_data, self.gt_path_data = path_data
        self.num_frames = num_frames
        self.transforms = transforms
        self.mode = mode
        self.map_idx_to_video = [{'input': os.path.join(self.input_path_data, folder), 
                                  'gt': os.path.join(self.gt_path_data, folder)}
                                 for folder in sorted(os.listdir(self.input_path_data))]
        
    def __len__(self):
        return len(self.map_idx_to_video)

    def __getitem__(self, idx):
        folder = self.map_idx_to_video[idx]
        frames = sorted([x for x in os.listdir(os.path.join(folder['input'], 'frames')) if '.png' in x])
        
        if self.mode == 'val':
            # Берем первый кадр для валидации
            start_idx = self.num_frames
        else:
            start_idx = np.random.randint(0, len(frames) - self.num_frames + 1)
        
        end_idx = start_idx + self.num_frames
        
        fragment = []
        for fname in frames[start_idx:end_idx]:
            frame = padding(
                        cv2.cvtColor(
                            cv2.imread(os.path.join(folder['input'], 'frames', fname)), 
                        cv2.COLOR_BGR2RGB)
                    ).astype('float32') / 255.
            fragment.append(self.transforms(frame))
        
        # Предсказываем карту внимания для последнего кадра в случайном подмножестве frames
        # Не забываем делать паддинг и нормализацию для единообразия
        saliency = normalize_map(padding(
                 cv2.imread(os.path.join(folder['gt'], 'gt_saliency', frames[end_idx - 1]), 
                 cv2.IMREAD_GRAYSCALE)))[np.newaxis].astype('float32')
        
        # Бейзлайн модель использует функцию потерь, основанную только на карте saliency
        # Но если понадобится, вы можете использовать и карты фиксаций
#         fixations = padding_fixation(
#                             cv2.imread(os.path.join(folder['gt'], 'gt_fixations', frames[end_idx - 1]), 
#                             cv2.IMREAD_GRAYSCALE))[np.newaxis].astype('float32')
        
        return fragment, saliency


# Бесконечное равномерное семплирование из датасета
class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self.size = size

    def _infinite_indices(self):
        g = torch.Generator()
        while True:
            yield from torch.randperm(self.size, generator=g)

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)
        
class DatasetClassWithOneObserver(DatasetClass):
    def __getitem__(self, idx):
        folder = self.map_idx_to_video[idx]
        frames = sorted([x for x in os.listdir(os.path.join(folder['input'], 'frames')) if '.png' in x])
        
        if self.mode == 'val':
            # Берем первый кадр для валидации
            start_idx = self.num_frames
            observer_idx = '00'
        else:
            start_idx = np.random.randint(0, len(frames) - self.num_frames + 1)
            observer_idx = np.random.choice(os.listdir(os.path.join(folder['input'], 'observers')))
        
        end_idx = start_idx + self.num_frames
        
        fragment = []
        observer_maps = []
        for fname in frames[start_idx:end_idx]:
            frame = padding(
                        cv2.cvtColor(
                            cv2.imread(os.path.join(folder['input'], 'frames', fname)), 
                        cv2.COLOR_BGR2RGB)
                    ).astype('float32') / 255.
            fragment.append(self.transforms(frame))
            
            observer_saliency = normalize_map(padding(
                 cv2.imread(os.path.join(folder['input'], 'observers', observer_idx,  'gaussians', frames[end_idx - 1]), 
                 cv2.IMREAD_GRAYSCALE)))[np.newaxis].astype('float32')
            observer_maps.append(observer_saliency)
        
        # Предсказываем карту внимания для последнего кадра в случайном подмножестве frames
        # Не забываем делать паддинг и нормализацию для единообразия
        saliency = normalize_map(padding(
                 cv2.imread(os.path.join(folder['gt'], 'gt_saliency', frames[end_idx - 1]), 
                 cv2.IMREAD_GRAYSCALE)))[np.newaxis].astype('float32')
        
        return fragment, observer_maps, saliency