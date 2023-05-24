import argparse
import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from segment_anything import SamPredictor, sam_model_registry
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
    device = torch.device('cuda')

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(torch.device('cuda'))

    sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    os.chdir(args.source)
    os.makedirs('../mask', exist_ok=True)

    for img_path in tqdm(os.listdir('.')):
        img = plt.imread(img_path)
        h, w, _ = img.shape
        if w > h:
            img_in = cv2.copyMakeBorder(img, 0, w - h, 0, 0, cv2.BORDER_CONSTANT)
        elif h > w:
            img_in = cv2.copyMakeBorder(img, 0, 0, 0, h - w, cv2.BORDER_CONSTANT)
        inputs = processor(text=prompt, images=[img_in], padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        logits = cv2.resize(logits, img_in.shape[:2])[:h, :w]
        logits = logits.flatten()
        ind = np.argsort(logits)[::-1]
        selected = ind[(np.linspace(0, 0.05, 4) * len(ind)).astype(int)]
        rows, cols = np.divmod(selected, w)
        pts = np.stack([cols, rows], axis=1)

        predictor.set_image(img)
        labels = np.ones(4, dtype=int)
        masks, _, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=False,
        )
        mask = masks[0]

        img = img * mask[..., None]
        plt.imsave(img_path, img)
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join('../mask', img_path), mask)
