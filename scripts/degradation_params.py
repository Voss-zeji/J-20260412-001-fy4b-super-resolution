# Auto-generated degradation parameters
# Calibrated: 2026-06-10
# Method: CH07 real paired data (2000M vs 4000M)
import numpy as np
from scipy.ndimage import gaussian_filter

PSF_SIGMA = 0.384
NOISE_STD = 0.4355

def degrade(hr_img):
    """Apply calibrated degradation to HR image for synthetic LR"""
    blurred = gaussian_filter(hr_img, sigma=PSF_SIGMA, mode='reflect')
    H, W = blurred.shape
    lr = blurred.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))
    noise = np.random.normal(0, NOISE_STD, lr.shape).astype(np.float32)
    return lr + noise
