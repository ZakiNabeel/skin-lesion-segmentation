import numpy as np

def calculate_dice_coefficient(ground_truth, predicted_mask):
    """
    Calculates the Dice Coefficient between the ground truth and predicted lesion mask.
    Both inputs should be binary numpy arrays (0s and 1s).
    """
    # Ensure both arrays are standard boolean arrays for logical operations
    gt_bool = ground_truth.astype(bool)
    pred_bool = predicted_mask.astype(bool)

    # True Positives (TP): Pixels correctly identified as lesion
    TP = np.sum(gt_bool & pred_bool)
    
    # False Positives (FP): Background pixels incorrectly identified as lesion
    FP = np.sum(~gt_bool & pred_bool)
    
    # False Negatives (FN): Lesion pixels incorrectly identified as background
    FN = np.sum(gt_bool & ~pred_bool)

    # Dice Coefficient Formula
    # Dice = (2 * TP) / (FN + (2 * TP) + FP)
    denominator = FN + (2 * TP) + FP
    
    # Handle edge case where both the ground truth and prediction are completely blank
    if denominator == 0:
        return 1.0 
        
    dice = (2.0 * TP) / denominator
    return dice