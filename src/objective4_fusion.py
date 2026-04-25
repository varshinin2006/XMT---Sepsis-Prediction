import numpy as np

def fuse_predictions(icu_pred, xray_pred):
    """
    icu_pred: probability (0-1)
    xray_pred: probability (0-1)
    """

    final_score = (0.7 * icu_pred) + (0.3 * xray_pred)

    if final_score > 0.5:
        return "HIGH SEPSIS RISK"
    else:
        return "LOW SEPSIS RISK"