import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from byol_pytorch import BYOL
import pytorch_lightning as pl
import glob
import random
# import wandb
import math
import json
from PIL import ImageFile   




ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_EXTS = ['.jpg', '.png', '.jpeg','.tif']
BATCH_SIZE = 1
NUM_GPUS   = 2
NUM_WORKERS = 2  #multiprocessing.cpu_count
n_label = 4
IMAGE_SIZE = 512


parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

img_paths = glob.glob('/workspace/Data/solo_train/*/*.tif')  #sftp://subrat@10.107.47.139 /workspace/Data/solo_train_copy/
csv_file = '/workspace/Data/Clean_train_data_encd.csv'


def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size, train):
        super().__init__()
        self.folder = folder
        self.paths = []
        self.image_size = image_size
        self.train = train
        csv_file="/workspace/Data/Clean_train_data_encd.csv"
        self.csv_data = pd.read_csv(csv_file)
        print(f" The lenghth of the csv is {len(self.csv_data)}")

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        self.transform = transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=1.5),
            # transforms.RandomRotation(degrees=(0, 135)),
            transforms.RandomAffine(degrees=(-30, 70), translate=(0, 0.15), scale=(0.75, 1)),
            transforms.Resize(math.floor(image_size*1.2)),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(contrast=0.65),
            # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.3)),# change from 9 to 5 and 2 to 0.5
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale),
            
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img = self.transform(img)
        img_name = str(path).split('/')[-1]
        label = self.csv_data.set_index('Name').loc[img_name, 'label']
        return img, img_name, label
# main

if __name__ == '__main__':
   
    ds_c = ImagesDataset(args.image_folder, IMAGE_SIZE, train = False)
    img_loader = DataLoader(ds_c, batch_size=1, num_workers=2, shuffle=False)
    count = 0
    for img, _, label in img_loader:
      ################################################  print(img.size())  #32 3 512 512
        img = np.transpose(img, (2, 3, 1, 0))
        img = img.reshape(512, 512, 3)
        img_NAME = '/workspace/Visualize/VV9wogblur_contrastlllllll/'+ 'aug_'+str(count)+'.png'
        plt.imshow(img)
        plt.savefig(img_NAME)
        count+=1
       



