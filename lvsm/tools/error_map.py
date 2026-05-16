import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 假设 gt, pred 已经是 H×W×3 的 float32 图
def gradient_error_map(gt, pred):
    # 转灰度
    gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pd_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    # Sobel
    gx_gt = cv2.Sobel(gt_g, cv2.CV_32F, 1, 0, ksize=3)
    gy_gt = cv2.Sobel(gt_g, cv2.CV_32F, 0, 1, ksize=3)
    gx_pd = cv2.Sobel(pd_g, cv2.CV_32F, 1, 0, ksize=3)
    gy_pd = cv2.Sobel(pd_g, cv2.CV_32F, 0, 1, ksize=3)
    # 幅值
    mag_gt = np.sqrt(gx_gt**2 + gy_gt**2)
    mag_pd = np.sqrt(gx_pd**2 + gy_pd**2)
    # 误差
    e = np.abs(mag_gt - mag_pd)
    # 归一化 [0,1]
    return (e - e.min()) / (e.max() - e.min())


# Load the combined image
img = cv2.imread('/home/qingwen/workspace/LVSM/experiments/evaluation/kubric_tab2_256_input7/cam2/000000/gt_vs_pred.png')
# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# h, w, _ = img.shape
# gt = img[:, :w//2]
# pred = img[:, w//2:]

# # 1. Pixel-wise Euclidean error map (L2 per pixel)
# diff = (gt.astype(float) - pred.astype(float))
# l2_error = np.linalg.norm(diff, axis=2)
# l2_norm = (l2_error - l2_error.min()) / (l2_error.max() - l2_error.min())

# # 2. Local SSIM map (multi-channel using channel_axis)
# _, ssim_map = ssim(
#     gt, 
#     pred, 
#     channel_axis=2, 
#     full=True
# )
# # Normalize SSIM to [0,1]
# ssim_norm = (ssim_map - ssim_map.min()) / (ssim_map.max() - ssim_map.min())

# # Display
# plt.figure(figsize=(8,4))
# plt.imshow(np.log(l2_norm), interpolation='bilinear')
# plt.title('Pixel-wise L2 Error Map')
# plt.axis('off')

# plt.figure(figsize=(8,4))
# plt.imshow(ssim_norm, interpolation='bilinear')
# plt.title('Local SSIM Map')
# plt.axis('off')

# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Load combined image
# img = cv2.imread('/mnt/data/5cd3de54-6003-47b6-9d4b-5deab75d99b7.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# h, w, _ = img.shape
# gt = img[:, :w//2].astype(np.float32)
# pred = img[:, w//2:].astype(np.float32)

# # Convert to grayscale
# gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
# pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)

# # 1. Gradient Error Map
# def gradient_error_map(g, p):
#     gx_g = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
#     gy_g = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
#     gx_p = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
#     gy_p = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
#     mag_g = np.sqrt(gx_g**2 + gy_g**2)
#     mag_p = np.sqrt(gx_p**2 + gy_p**2)
#     e = np.abs(mag_g - mag_p)
#     return (e - e.min()) / (e.max() - e.min())

# grad_err = gradient_error_map(gt_gray, pred_gray)

# # 2. Laplacian Error Map
# lap_gt = cv2.Laplacian(gt_gray, cv2.CV_32F)
# lap_pred = cv2.Laplacian(pred_gray, cv2.CV_32F)
# lap_err = np.abs(lap_gt - lap_pred)
# lap_err_norm = (lap_err - lap_err.min()) / (lap_err.max() - lap_err.min())

# # 3. Local Variance Ratio Map
# win = 11
# eps = 1e-6
# # Compute local mean of image and squared image
# mean_gt = cv2.blur(gt_gray, (win, win))
# mean_pred = cv2.blur(pred_gray, (win, win))
# mean_sq_gt = cv2.blur(gt_gray**2, (win, win))
# mean_sq_pred = cv2.blur(pred_gray**2, (win, win))
# # Variance = E[x^2] - E[x]^2
# var_gt = np.maximum(mean_sq_gt - mean_gt**2, 0)
# var_pred = np.maximum(mean_sq_pred - mean_pred**2, 0)
# ratio = 1 - var_pred / (var_gt + eps)
# var_ratio_norm = (ratio - ratio.min()) / (ratio.max() - ratio.min())

# # Plotting
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# axs[0].imshow(grad_err, interpolation='bilinear')
# axs[0].set_title('Gradient Error Map')
# axs[1].imshow(lap_err_norm, interpolation='bilinear')
# axs[1].set_title('Laplacian Error Map')
# axs[2].imshow(var_ratio_norm, interpolation='bilinear')
# axs[2].set_title('Local Variance Ratio Map')

# for ax in axs:
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

h, w, _ = img.shape
gt = img[:, :w//2].astype(np.float32)
pred = img[:, w//2:].astype(np.float32)

# Parameters
win = 3  # sliding window size for local PSNR
C = 3     # number of channels
MAX_I = 255.0

# 1. Compute per-pixel squared error sum across channels
sq_err = np.sum((gt - pred)**2, axis=2)

# 2. Compute local MSE by box-filter over a window
kernel = np.ones((win, win), dtype=np.float32)
local_mse = cv2.filter2D(sq_err, -1, kernel) / (win * win * C)

# Avoid division by zero
local_mse[local_mse == 0] = 1e-10

# 3. Local PSNR map
psnr_map = 10 * np.log10((MAX_I**2) / local_mse)

# 4. Normalize for visualization (invert: high error -> bright)
psnr_vis = (psnr_map.max() - psnr_map)  # lower PSNR = higher error
psnr_vis = (psnr_vis - psnr_vis.min()) / (psnr_vis.max() - psnr_vis.min())

# Display
plt.figure(figsize=(8,4))
plt.imshow(np.exp(psnr_vis), cmap='viridis', interpolation='bilinear')
plt.title('Local PSNR Error Map')
plt.axis('off')
plt.show()