import sys
import os

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from icpReconstructor.torch_reconstruction import TorchCurveEstimator, TorchRotationFrame, TorchPolynomialCurve, Polynomial3Torch, image_to_idx, camera_folder_to_params, PolynomialKTorch
from skimage.morphology import disk, binary_dilation
from icpReconstructor.utils import fromWorld2Img
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time

#%% Initialization
l = torch.tensor([0.0750, 0.1300, 0.1900])  # length of the segments

"""
Load camera calibration files and simulate one set of cannulas.
"""
camera_folder = "camera_calibration_files"

cam_params = camera_folder_to_params(camera_folder, 2)

A0 = cam_params[0]["A"]
A1 = cam_params[1]["A"]
dist0 = cam_params[0]["dist"] # distortion parameters
dist1 = cam_params[1]["dist"]
P0 = cam_params[0]["P"]
P1 = cam_params[1]["P"]

R = cam_params[0]["R"]
T = cam_params[0]["T"]

plot_curvature = True
n_iter = 8

#%% Image data
"""
    Project the resulting cannula onto the image planes and generate two images by thickening the line
"""

im0_bin = plt.imread("im0.png") > 0
im1_bin = plt.imread("im1.png") > 0
p0_img = image_to_idx(im0_bin)
p1_img = image_to_idx(im1_bin)

#%%
now = time()
curve = TorchRotationFrame(l, integrator="rk4", rotation_method="rotm")
tce = TorchCurveEstimator(curve, cam_params, l[-1], w=0.0, n_steps=40, dist_norm=2)
if n_iter > 0:
    optimizer = torch.optim.Adam(tce.parameters(), 0.2, weight_decay=0)
    loss_hist = tce.fit([p0_img, p1_img], optimizer, n_iter=n_iter, batch_size=4000, repetitions=1)

print(f"The algorithm ran {time()-now} s")

curve.integrator = "dopri5"

s = torch.linspace(0, l[-1], 1000)
p_out = curve(s).detach().numpy()

# %%
p_img0 = fromWorld2Img(torch.from_numpy(p_out[:, :3]).T, A0, dist0, P0, R, T)
p_img1 = fromWorld2Img(torch.from_numpy(p_out[:, :3]).T, A1, dist1, P1, R, T)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.plot(p_out[:, 0], p_out[:, 1], p_out[:, 2], linewidth=4, label='reconstruction')
ax1.axis('equal')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax1.legend(fontsize=20)
if n_iter > 0:
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('Epoch')
    ax2.plot(loss_hist)
    ax2.plot(np.argmin(loss_hist), np.min(loss_hist), 'x', markersize=8, markeredgewidth=4, label="min loss")
    ax2.title.set_text('Loss')
    ax2.title.set_fontsize(20)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', labelsize=16)
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(im0_bin)
ax3.title.set_text('Binary Image 0')
ax3.plot(p_img0[0, :], p_img0[1, :], label="reconstruction")
ax3.legend(fontsize=20)
ax3.set_xticks([])
ax3.set_yticks([])

ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(im1_bin)
ax4.title.set_text('Binary Image 1')
ax4.plot(p_img1[0, :], p_img1[1, :], label="reconstruction")
ax4.legend(fontsize=20)
ax4.set_xticks([])
ax4.set_yticks([])

if plot_curvature:
    fig2 = plt.figure()
    ax5 = fig2.add_subplot(311)
    ax6 = fig2.add_subplot(312)
    ax7 = fig2.add_subplot(313)
    s_pl = s[s <= curve.l[0]]
    ax5.plot(s_pl, curve.ux_funs[0](s_pl).detach().numpy())
    ax6.plot(s_pl, curve.uy_funs[0](s_pl).detach().numpy())
    ax7.plot(s_pl, curve.uz_funs[0](s_pl).detach().numpy())
    for i in range(1, 3):
        s_pl = s[(s <= curve.l[i]) & (curve.l[i-1] <= s)]
        ax5.plot(s_pl, curve.ux_funs[i](s_pl).detach().numpy())
        ax6.plot(s_pl, curve.uy_funs[i](s_pl).detach().numpy())
        ax7.plot(s_pl, curve.uz_funs[i](s_pl).detach().numpy())

plt.show()
