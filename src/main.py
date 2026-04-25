import os
import cv2
import numpy as np
import pandas as pd
from segmenters import (
    adaptive_thresholding, 
    kmeans_segmentation_numpy, 
    canny_segmentation, 
    marr_hildreth_segmentation, 
    manual_combination_segmentation
)
from evaluation import calculate_dice_coefficient

# Define Paths
ORIGINAL_IMAGES_DIR = '../data/Original Images/'
GROUND_TRUTH_DIR = '../data/Ground Truths/'
OUTPUT_DIR = '../output/'

# Ensure output directories exist
algorithms = ['Adaptive', 'KMeans', 'Canny', 'Marr_Hildreth', 'Manual']
for algo in algorithms:
    os.makedirs(os.path.join(OUTPUT_DIR, algo), exist_ok=True)

def main():
    results = []
    
    # Get list of all images
    image_files = sorted([f for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.endswith('.bmp')])
    
    for img_name in image_files:
        print(f"Processing {img_name}...")
        
        # Load Original Image
        img_path = os.path.join(ORIGINAL_IMAGES_DIR, img_name)
        image = cv2.imread(img_path)
        
        # Load corresponding Ground Truth
        gt_name = img_name.replace('.bmp', '_lesion.bmp')
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_name)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize Ground Truth (0 and 1)
        gt_binary = (gt_image > 127).astype(np.uint8)
        
        # --- Apply Algorithms ---
        masks = {
            'Adaptive': adaptive_thresholding(image),
            'KMeans': kmeans_segmentation_numpy(image, k=2, max_iters=20),
            'Canny': canny_segmentation(image),
            'Marr_Hildreth': marr_hildreth_segmentation(image),
            'Manual': manual_combination_segmentation(image)
        }
        
        # Calculate Dice and Save Output Masks
        row_data = {'Image': img_name}
        for algo_name, pred_mask in masks.items():
            dice_score = calculate_dice_coefficient(gt_binary, pred_mask)
            row_data[algo_name] = round(dice_score, 4)
            
            # Save edge map (multiply by 255 to make it visible)
            output_mask_path = os.path.join(OUTPUT_DIR, algo_name, img_name)
            cv2.imwrite(output_mask_path, pred_mask * 255)
            
        results.append(row_data)
        
    # Create Report Table
    df = pd.DataFrame(results)
    
    # Calculate Averages for the last row
    averages = df.mean(numeric_only=True).round(4)
    avg_row = pd.DataFrame([{'Image': 'AVERAGE', **averages.to_dict()}])
    df = pd.concat([df, avg_row], ignore_index=True)
    
    # Save the table for the PDF report
    df.to_csv('../report/dice_scores.csv', index=False)
    print("\nProcessing Complete! Check the 'report' folder for dice_scores.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()