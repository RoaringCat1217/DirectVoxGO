from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import os
import cv2
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    args = parser.parse_args()

    print('loading model...')
    device = torch.device('cuda')
    sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    os.chdir(args.source)
    os.makedirs('../mask', exist_ok=True)

    for img_path in tqdm(os.listdir('.')):
        img = plt.imread(img_path)
        masks = mask_generator.generate(img)
        mask = masks[0]['segmentation']
        img = img * mask[..., None]
        plt.imsave(img_path, img)
        plt.imsave(os.path.join('../mask', img_path), mask)
