import numpy as np
import cv2
from collections import Counter
import os
from glob import glob
import cv2
import sys
import pandas as pd
import time
from argparse import ArgumentParser
import numpy as np
from collections import Counter
from PIL import Image
import os
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import generate_binary_structure, gaussian_filter

def apply_gaussian_filter_to_mask(mask, sigma):
    filtered_mask = gaussian_filter(mask.astype(float), sigma)
    return (filtered_mask > 0.5).astype(np.uint8)

def modify_structure_size(scale):
    return generate_binary_structure(2, 2) * scale

def randomize_contour(image, morph_prob, max_iterations, structure_scale, sigma):
    unique_masks = np.unique(image)
    modified_image = np.zeros_like(image)

    struct = modify_structure_size(structure_scale)

    for mask_value in unique_masks:
        if mask_value == 0:
            continue
        
        mask = (image == mask_value).astype(np.uint8)
        
        # Find initial boundaries by dilation and erosion
        dilated = binary_dilation(mask, structure=struct)
        eroded = binary_erosion(mask, structure=struct)
        boundary = dilated ^ eroded

        for _ in range(max_iterations):
            changes = np.random.rand(*boundary.shape) < morph_prob
            if np.random.rand() > 0.5:
                boundary = binary_dilation(boundary, mask=changes, structure=struct)
            else:
                boundary = binary_erosion(boundary, mask=changes, structure=struct)
        
        # Apply the randomized boundary back to the mask
        modified_mask = np.where(boundary, 1, mask)
        modified_mask = apply_gaussian_filter_to_mask(modified_mask, sigma)
        modified_image = np.where(modified_mask, mask_value, modified_image)

    return modified_image

def process_masks(mask, kernel_size, iter, morph_operation):
    # Get unique labels from the mask
    labels = np.unique(mask)[1:]  # Exclude the background label (0)

    # Create an empty mask for storing individual binary masks
    binary_masks = np.zeros_like(mask, dtype=np.uint8)

    for label in labels:
        # Create a binary mask for the current label
        label_mask = np.uint8(mask == label)

        # Connect disconnected parts of the current label using the specified morphological operation
        connected_mask = connect_disconnected_parts(label_mask, kernel_size, iter, morph_operation)

        # Assign the connected parts back to the original mask for the current label
        binary_masks[connected_mask != 0] = label

    return binary_masks

def connect_disconnected_parts(mask, kernel_size, iter, morph_operation):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if morph_operation == 'E':
        connected_mask = cv2.erode(mask, kernel, iterations=iter)
    elif morph_operation == 'D':
        connected_mask = cv2.dilate(mask, kernel, iterations=iter)
        
    elif morph_operation == 'EAD':
        dilated_mask = cv2.dilate(mask, kernel, iterations=iter)
        connected_mask = cv2.erode(dilated_mask, kernel, iterations=iter)
    elif morph_operation == 'DAE':
        eroded_mask = cv2.erode(mask, kernel, iterations=iter)
        connected_mask = cv2.dilate(eroded_mask, kernel, iterations=iter)
    else:
        raise ValueError("Invalid morphological operation specified")

    return connected_mask

def remove_255(mask):
    mask[mask == 255] = 0
    return mask

def check_mask_values(mask):
    if np.any(mask >= 11):
        print('mask : ', Counter(mask.flatten()))
        
def check_processed_mask_values(mask):
    if np.any(mask >= 11):
        print('processed_mask : ', Counter(mask.flatten()))

def apply_palette_to_mask(original_mask_path, processed_mask):
    palette = Image.open(original_mask_path).getpalette()
    processed_mask_img = Image.fromarray(processed_mask.astype(np.uint8))
    processed_mask_img.putpalette(palette)
    return processed_mask_img

def apply_color_to_mask(processed_mask, original_mask):
    colored_mask = np.zeros(original_mask.shape, dtype=np.uint8)

    labels = np.unique(processed_mask)

    for label in labels:
        if label == 0:  # Ignore the background
            continue

        original_color = np.median(original_mask[processed_mask == label], axis=0)
        colored_mask[processed_mask == label] = original_color

    return colored_mask

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    # Check if the union is empty
    if np.sum(union) == 0:
        iou = 1.0
    else:
        iou = np.sum(intersection) / np.sum(union)
    # Check if the result is NaN
    if np.isnan(iou):
        raise ValueError("IoU is NaN. Check your input masks.")
    return iou

def jaccard(main_folder_1, main_folder_2):
    subfolders_1 = os.listdir(main_folder_1)
    subfolders_2 = os.listdir(main_folder_2)

    total_iou = 0
    total_comparisons = 0
    
    for subfolder in tqdm(subfolders_1 if len(subfolders_1) < len(subfolders_2) else subfolders_2,
                          desc="Comparing video folders"):
        subfolder_path_1 = os.path.join(main_folder_1, subfolder)
        subfolder_path_2 = os.path.join(main_folder_2, subfolder)

        mask_files = [f for f in os.listdir(subfolder_path_1) if f.endswith('.png')]

        for mask_file in mask_files:
            mask_path_1 = os.path.join(subfolder_path_1, mask_file)
            mask_path_2 = os.path.join(subfolder_path_2, mask_file)

            mask1 = np.array(Image.open(mask_path_1).convert('L'))
            mask2 = np.array(Image.open(mask_path_2).convert('L'))

            mask1 = np.where(mask1 >= 128, 1, 0)
            mask2 = np.where(mask2 >= 128, 1, 0)

            iou = calculate_iou(mask1, mask2)

            total_iou += iou
            total_comparisons += 1

    average_iou = total_iou / total_comparisons
    print(f"Average IoU: {average_iou}")