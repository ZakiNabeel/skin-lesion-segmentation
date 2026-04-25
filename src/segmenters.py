import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# HELPER FUNCTIONS (PURE NUMPY)
# ==========================================

def bgr2gray(image):
    """Converts BGR image (loaded via cv2) to Grayscale."""
    return np.dot(image[..., :3], [0.1140, 0.5870, 0.2989])

def convolve2d(image, kernel):
    """Fast 2D convolution using SIMD-friendly sliding windows."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    windows = sliding_window_view(padded, (kh, kw))
    return np.sum(windows * kernel, axis=(2, 3))

def morph_dilate(binary_image, size=5):
    """Mathematical morphology: Dilation."""
    pad = size // 2
    padded = np.pad(binary_image, pad, mode='edge')
    windows = sliding_window_view(padded, (size, size))
    return np.max(windows, axis=(2, 3))

def morph_erode(binary_image, size=5):
    """Mathematical morphology: Erosion."""
    pad = size // 2
    padded = np.pad(binary_image, pad, mode='edge')
    windows = sliding_window_view(padded, (size, size))
    return np.min(windows, axis=(2, 3))

def remove_hair_numpy(gray_image):
    """
    Pure NumPy implementation of DullRazor-like hair removal.
    Isolates dark thin structures (hairs) and inpaints them with the background.
    """
    # Closing removes dark hairs
    closed = morph_erode(morph_dilate(gray_image, 9), 9)
    # Difference highlights only the hairs
    hairs = closed - gray_image
    hair_mask = hairs > 15
    # Replace hairs with the smoothed background
    return np.where(hair_mask, closed, gray_image)

# ==========================================
# 1. K-MEANS CLUSTERING (CUSTOM)
# ==========================================

def kmeans_segmentation_numpy(image, k=2, max_iters=20):
    h, w, c = image.shape
    pixels = image.reshape((-1, c)).astype(np.float32)
    np.random.seed(42)
    
    centroids = pixels[np.random.choice(pixels.shape[0], size=k, replace=False)]

    for _ in range(max_iters):
        distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            pixels[labels == i].mean(axis=0) if len(pixels[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    segmented = labels.reshape((h, w))
    # Melanoma/Nevus are darker than skin; select the darkest cluster
    lesion_cluster = np.argmin(centroids.mean(axis=1))
    return (segmented == lesion_cluster).astype(np.uint8)

# ==========================================
# 2. ADAPTIVE THRESHOLDING (CUSTOM)
# ==========================================

def adaptive_thresholding(image, block_size=21, C=5):
    # Apply hair removal pre-processing
    clean_gray = remove_hair_numpy(bgr2gray(image))
    
    kernel = np.ones((block_size, block_size)) / (block_size ** 2)
    local_mean = convolve2d(clean_gray, kernel)
    
    mask = (clean_gray < (local_mean - C)).astype(np.uint8)
    return morph_dilate(morph_erode(mask, 5), 5)

# ==========================================
# 3. CANNY EDGE DETECTOR (CUSTOM)
# ==========================================

def canny_segmentation(image, low_thresh=20, high_thresh=60):
    # Apply hair removal pre-processing
    clean_gray = remove_hair_numpy(bgr2gray(image))
    
    gauss = np.array([[2,4,5,4,2], [4,9,12,9,4], [5,12,15,12,5], [4,9,12,9,4], [2,4,5,4,2]]) / 159.0
    blurred = convolve2d(clean_gray, gauss)
    
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = convolve2d(blurred, Kx)
    Iy = convolve2d(blurred, Ky)
    
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    angle = np.rad2deg(np.arctan2(Iy, Ix)) % 180
    
    quantized = np.zeros_like(angle)
    quantized[(angle < 22.5) | (angle >= 157.5)] = 0
    quantized[(angle >= 22.5) & (angle < 67.5)] = 45
    quantized[(angle >= 67.5) & (angle < 112.5)] = 90
    quantized[(angle >= 112.5) & (angle < 157.5)] = 135

    G_up, G_down = np.roll(G, -1, axis=0), np.roll(G, 1, axis=0)
    G_left, G_right = np.roll(G, -1, axis=1), np.roll(G, 1, axis=1)
    G_ul, G_dr = np.roll(G_up, -1, axis=1), np.roll(G_down, 1, axis=1)
    G_ur, G_dl = np.roll(G_up, 1, axis=1), np.roll(G_down, -1, axis=1)

    mask0 = (quantized == 0) & (G >= G_left) & (G >= G_right)
    mask45 = (quantized == 45) & (G >= G_ur) & (G >= G_dl)
    mask90 = (quantized == 90) & (G >= G_up) & (G >= G_down)
    mask135 = (quantized == 135) & (G >= G_ul) & (G >= G_dr)

    NMS = G * (mask0 | mask45 | mask90 | mask135)
    
    edges = np.zeros_like(NMS, dtype=np.uint8)
    strong = NMS >= high_thresh
    weak = (NMS >= low_thresh) & (NMS < high_thresh)
    edges[strong] = 1
    
    for _ in range(3): 
        strong_dilated = morph_dilate(edges, 3)
        edges[weak & (strong_dilated == 1)] = 1
        weak = weak & ~(edges == 1)
        
    return morph_dilate(edges, 15)

# ==========================================
# 4. MARR-HILDRETH (CUSTOM)
# ==========================================

def marr_hildreth_segmentation(image):
    # Apply hair removal pre-processing
    clean_gray = remove_hair_numpy(bgr2gray(image))
    
    log_kernel = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ])
    log_img = convolve2d(clean_gray, log_kernel)
    
    padded = np.pad(log_img, 1, mode='edge')
    windows = sliding_window_view(padded, (3, 3))
    max_vals = np.max(windows, axis=(2, 3))
    min_vals = np.min(windows, axis=(2, 3))
    
    zero_crossing = ((log_img > 0) & (min_vals < 0)) | ((log_img < 0) & (max_vals > 0))
    variance = np.var(windows, axis=(2, 3))
    
    edges = (zero_crossing & (variance > np.mean(variance) * 0.5)).astype(np.uint8)
    return morph_dilate(edges, 15)

# ==========================================
# 5. MANUAL COMBINATION (CUSTOM OTSU)
# ==========================================

def manual_combination_segmentation(image):
    # Apply hair removal pre-processing
    clean_gray = remove_hair_numpy(bgr2gray(image))
    
    hist, _ = np.histogram(clean_gray.ravel(), 256, [0, 256])
    total = clean_gray.size
    
    current_max, threshold = 0, 0
    sumT = np.dot(np.arange(256), hist)
    sumB, weightB = 0.0, 0.0
    
    for i in range(256):
        weightB += hist[i]
        if weightB == 0: continue
        weightF = total - weightB
        if weightF == 0: break
        
        sumB += i * hist[i]
        mB = sumB / weightB
        mF = (sumT - sumB) / weightF
        
        varBetween = weightB * weightF * (mB - mF) ** 2
        if varBetween > current_max:
            current_max = varBetween
            threshold = i
            
    mask = (clean_gray < threshold).astype(np.uint8)
    return morph_erode(morph_dilate(mask, 15), 15)