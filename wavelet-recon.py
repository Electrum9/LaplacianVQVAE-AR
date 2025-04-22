import torch
import pywt
import ptwt
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.util import img_as_float
import numpy as np

# Load and preprocess grayscale image
img = img_as_float(data.camera())
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

# Define wavelet
wavelet = pywt.Wavelet('db8')

# 2D DWT (1 level)
coeffs = ptwt.wavedec2(img_tensor, wavelet, level=1, mode='zero')
LL, (LH, HL, HH) = coeffs

# Convert to numpy for plotting
LL_np = LL.squeeze().numpy()
LH_np = LH.squeeze().numpy()
HL_np = HL.squeeze().numpy()
HH_np = HH.squeeze().numpy()

# Normalize subbands for visualization
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

LL_img = normalize(LL_np)
LH_img = normalize(np.abs(LH_np))
HL_img = normalize(np.abs(HL_np))
HH_img = normalize(np.abs(HH_np))

# Plot
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(LL_img, cmap='gray')
plt.title("LL (Approximation)")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(LH_img, cmap='gray')
plt.title("LH (Vertical Detail)")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(HL_img, cmap='gray')
plt.title("HL (Horizontal Detail)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(HH_img, cmap='gray')
plt.title("HH (Diagonal Detail)")
plt.axis('off')

plt.tight_layout()
plt.savefig('wavelet.png')
