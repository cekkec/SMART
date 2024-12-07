import os
from glob import glob
from noise_tool import *  # Assuming the function is in a file named process_masks.py
# from SLIC import superpixel_segmentation_slic
import cv2
import sys
import pandas as pd
import time
from argparse import ArgumentParser
import numpy as np
from collections import Counter
from PIL import Image
import os
from tqdm import tqdm  # tqdm 라이브러리를 불러옵니다.

parser = ArgumentParser()
parser.add_argument('--dataset', help='D17/YTV', default='D17')
parser.add_argument('--d17_path', default='../DAVIS/2017')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised')
parser.add_argument('--split', help='val/test', default='train')

# Mopology
parser.add_argument('--mode', type=str, help='DE:Dilation+Erosion / Mar:Markov Random Contour', default='DE')
parser.add_argument('--mopology', type=str, help='D:dilation / E:erosion / DAE:Erision->Dilation / EAD:Dilation->Erision', default='EAD')
parser.add_argument('--kernel_size', type=int, help='Dilation or Erosion kernel size', default=5)
parser.add_argument('--iterations', type=int, help='1~5', default=3)

# Mar
parser.add_argument('--morph_prob', help='dilation and erosion prob', default=0.5)
parser.add_argument('--max_iterations', help='max_iterations', default=30)
parser.add_argument('--structure_scale', help='structure_scale', default=3)
parser.add_argument('--sigma', help='sigma', default=3.0)
args = parser.parse_args()

is_davis = args.dataset.startswith('D')
is_youtube = args.dataset.startswith('Y')

mode = args.mode
# DE
morph = args.mopology
kernel = args.kernel_size
iter = args.iterations

# Mar
morph_prob = args.morph_prob
max_iterations = args.max_iterations
structure_scale = args.structure_scale
sigma = args.sigma

SET = args.split
TASK = args.task

noised_data = '{}_{}_{}_ker{}_it{}'.format(mode, args.dataset, morph, kernel, iter)
print('noised_data : ', noised_data)

if is_youtube:
    base_data = '../YouTube/train_480p/Annotations'
    output_path = f'../{noised_data}'

elif is_davis:
    base_data = '../DAVIS/2017/trainval/Annotations/480p'
    output_path = f'../{noised_data}'
    exclude_txt_path = '../DAVIS/2017/trainval/ImageSets/2017/val.txt'


def process_all_masks(mode, base_data, output_path, kernel, iterations, morph_operation,
                      morph_prob, max_iterations, structure_scale, sigma, exclude_txt_path=None):
    # Load the names of videos to be excluded
    if exclude_txt_path:
        with open(exclude_txt_path, 'r') as file:
            exclude_video_names = file.read().splitlines()
        
        video_folders = [f for f in glob(os.path.join(base_data, '*')) if os.path.isdir(f) and os.path.basename(f) not in exclude_video_names]
    
    else:
        video_folders = [f for f in glob(os.path.join(base_data, '*')) if os.path.isdir(f)]

    for video_folder in tqdm(video_folders, desc="Processing videos", unit="video"):
        png_files = glob(os.path.join(video_folder, '*.png'))

        for png_file in png_files:
            mask = Image.open(png_file).convert('P')
            mask = np.array(mask, dtype=np.uint8)
            
            check_mask_values(mask)

            if mode == 'DE':
                processed_mask = process_masks(mask, kernel, iterations, morph_operation)
            elif mode == 'Mar':
                processed_mask = randomize_contour(mask, morph_prob, max_iterations, structure_scale, sigma)   

            processed_mask = remove_255(processed_mask)
            check_processed_mask_values(processed_mask)

            processed_mask_img = apply_palette_to_mask(png_file, processed_mask)

            relative_path = os.path.relpath(png_file, base_data)
            output_file_path = os.path.join(output_path, relative_path)
            
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            processed_mask_img.save(output_file_path)
            
if is_youtube:
    process_all_masks(mode, base_data, output_path, kernel, iter, morph,
                      morph_prob, max_iterations, structure_scale, sigma)
    jaccard(base_data, output_path)

if is_davis:
    process_all_masks(mode, base_data, output_path, kernel, iter, morph,
                      morph_prob, max_iterations, structure_scale, sigma, exclude_txt_path)