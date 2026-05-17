# 🔬 Skin Lesion Segmentation in Dermoscopy Imaging

A from-scratch image processing pipeline for automated skin lesion segmentation — built entirely with **pure NumPy** (no OpenCV, no scikit-image for core logic). Evaluated across 49 dermoscopic images using the Dice Coefficient.
<img width="645" height="335" alt="image" src="https://github.com/user-attachments/assets/3e9a0163-52e4-4f36-b497-ab6d9e10bdf3" />


---

## 📁 Project Structure

```
├── src/
│   ├── main.py          # Pipeline runner — processes all images & exports results
│   ├── segmenters.py    # All 5 segmentation algorithms + helper functions
│   └── evaluation.py   # Dice Coefficient calculator
├── data/
│   ├── Original Images/ # Input .bmp dermoscopy images
│   └── Ground Truths/   # Expert-annotated binary masks (_lesion.bmp)
├── output/
│   ├── Adaptive/
│   ├── KMeans/
│   ├── Canny/
│   ├── Marr_Hildreth/
│   └── Manual/
└── report/
    └── dice_scores.csv  # Per-image Dice scores for all 5 algorithms
```

---

## ⚙️ How It Works

### Pre-Processing: Hair Removal (DullRazor Concept)
Before segmentation, a custom mathematical hair removal step cleans the image:
1. Morphological closing (dilation → erosion) subsumes thin hair strands
2. Difference masking isolates the hair pixels
3. Inpainting replaces hair pixels with the smoothed background

This step improved Canny's average Dice score by ~47%.

### Segmentation Algorithms

| # | Algorithm | Avg. Dice | Approach |
|---|-----------|-----------|----------|
| 1 | Adaptive Thresholding | 0.0351 | Local neighborhood mean |
| 2 | K-Means Clustering | 0.6901 | Global color clustering |
| 3 | Canny Edge Detector | 0.3040 | Gradient + NMS + Hysteresis |
| 4 | Marr-Hildreth | 0.3228 | Laplacian of Gaussian zero-crossings |
| 5 | Manual Combination (Otsu) | **0.6957** | Global variance thresholding + morphology |

**Best performer: Manual Combination (Otsu's method)** at an average Dice of 0.6957.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas opencv-python
```

> Note: OpenCV is used **only** for image I/O (`cv2.imread` / `cv2.imwrite`). All algorithmic logic is pure NumPy.

### Run the Pipeline
```bash
# From the src/ directory
python main.py
```

This will:
- Process all `.bmp` images in `data/Original Images/`
- Save segmentation masks to `output/<algorithm>/`
- Export Dice scores to `report/dice_scores.csv`

---

## 📊 Key Results

Tested on **49 dermoscopic images** from the ISIC-style dataset.

```
Algorithm         | Avg Dice
------------------|----------
Adaptive          | 0.0351   ← worst (too sensitive to local noise)
KMeans            | 0.6901
Canny             | 0.3040
Marr-Hildreth     | 0.3228
Manual (Otsu)     | 0.6957   ← best
```

Region-based methods (K-Means, Otsu) significantly outperform edge-based methods for dermoscopy because lesions are defined by **global pigmentation**, not sharp boundaries.

---

## 🧠 Technical Highlights

- **Custom 2D Convolution** using `numpy.lib.stride_tricks.sliding_window_view` — no `scipy.signal`
- **Custom Morphological Operations** — dilation and erosion built from scratch
- **Full Canny Pipeline** — Gaussian blur → Sobel gradients → NMS → Hysteresis, all vectorized
- **Custom Otsu's Thresholding** — exhaustive inter-class variance search in pure NumPy
- **Custom K-Means** — Euclidean distance clustering with convergence detection

---

## 📄 Report

Full methodology, algorithmic flow diagrams, and per-image results are documented in `report/DIP_Assignment2_Report.pdf`.

---

## 👤 Author

**Zaki Nabeel** — Roll No: 23i-0508  
FAST-NUCES Islamabad | BS Computer Science  
[GitHub](https://github.com/ZakiNabeel) | [LinkedIn](https://www.linkedin.com/in/zakinabeel)
