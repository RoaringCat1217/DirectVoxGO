import argparse
import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs='+')
    parser.add_argument("--source", type=str)
    args = parser.parse_args()
    prompt = ' '.join(args.prompt)

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    os.chdir(args.source)
    os.makedirs('../mask', exist_ok=True)
    for img_path in tqdm(os.listdir('.')):
        img = plt.imread(img_path)
        h, w, _ = img.shape
        if w > h:
            img = cv2.copyMakeBorder(img, 0, w - h, 0, 0, cv2.BORDER_CONSTANT)
        elif h > w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, h - w, cv2.BORDER_CONSTANT)
        inputs = processor(text=prompt, images=[img], padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        mask = (outputs.logits > 0).numpy().astype(np.uint8)
        mask = cv2.resize(mask, img.shape[:2])
        img = img[:h, :w]
        mask = mask[:h, :w]
        img = img * mask[..., None]
        plt.imsave(img_path, img)
        plt.imsave(os.path.join('../mask', img_path), mask)
