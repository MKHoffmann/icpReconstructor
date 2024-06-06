#%%
import sys
import os

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from icpReconstructor.torch_reconstruction import TorchMovingFrame, PixelDataset
from icpReconstructor.casadi_reconstruction import CasadiCurveEstimator, Polynomial3Casadi, CasadiMovingFrame
from icpReconstructor.utils import fromWorld2Img, image_to_idx, camera_folder_to_params, PixelDataset
import torch
import matplotlib.pyplot as plt
from time import time
from random import sample

#%% Initialization
l = torch.tensor([0.0750, 0.1300, 0.1900])  # length of the reconstruction segments

"""
Load camera calibration files and simulate one set of cannulas.
"""
camera_folder = "camera_calibration_files"

cam_params = camera_folder_to_params(camera_folder, 2)
cam_params_cas = camera_folder_to_params(camera_folder, 2, package="casadi")

plot_curvature = True
n_iter = 8

#%% Image data
"""
    Load the images, binarize them and load the indices of the pixels that are part of the cannula.
"""

im0_bin = plt.imread("im0.png") > 0
im1_bin = plt.imread("im1.png") > 0
p0_img = image_to_idx(im0_bin)
p1_img = image_to_idx(im1_bin)

dataset = PixelDataset([p0_img, p1_img])

#%% epipolar line matching
now = time()

tip_estimator_params_0 = camera_folder + "param_cam_0.mat"
tip_estimator_params_1 = camera_folder + "param_cam_1.mat"

bin_threshold = 200

#%%

curvature_options = {"continuous":False, "random_init":False, "end_no_curvature":False}

n = len(l)
start = time()
model = CasadiMovingFrame(l.detach().numpy(), n_steps=40)
ux = Polynomial3Casadi(model.opt, n, **curvature_options)
uy = Polynomial3Casadi(model.opt, n, **curvature_options)
model.ux = ux
model.uy = uy

model.add_u_constraint("x", -40, 40)
model.add_u_constraint("y", -40, 40)

cce = CasadiCurveEstimator(model, cam_params_cas, l[-1].detach().numpy(), dist_norm=2)

cce.initial_solve()

frac = 2
loss_hist = []
dataset_idc = sample(range(len(dataset)), int(len(dataset)/frac))
for j in range(n_iter):
    
    cce._set_warmstart_ipopt(1e-6)
    cost = cce.icp_step(dataset.p[dataset_idc,:], dataset.img_idx_data[dataset_idc])
    loss_hist.append(cost)

print(f"The algorithm ran for {time() - start} s.")

ux = cce.sol.value(model.ux.u_p)
uy = cce.sol.value(model.uy.u_p)
uz = cce.sol.value(model.uz.u_p)
ux = torch.from_numpy(ux).float()
uy = torch.from_numpy(uy).float()
uz = torch.from_numpy(uz).float()

curvature = TorchMovingFrame(l, integrator="dopri5", rotation_method="rotm", ux=ux, uy=uy, uz=uz)
s = torch.linspace(0, l[-1], 1000)
p_out = curvature.integrate(s).detach().numpy()
    
#%%
p_img0 = fromWorld2Img(torch.from_numpy(p_out[:, :3]).T, **cam_params[0])
p_img1 = fromWorld2Img(torch.from_numpy(p_out[:, :3]).T, **cam_params[1])

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.plot(p_out[:, 0], p_out[:, 1], p_out[:, 2], linewidth=4, label='reconstruction')
ax1.axis('equal')
ax1.legend()
if n_iter > 0:
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('Epoch')
    ax2.semilogy(loss_hist)
    ax2.semilogy(torch.Tensor([loss_hist]).argmin(), torch.Tensor([loss_hist]).min(), 'x', markeredgewidth=2, label="min loss")
    ax2.title.set_text('Loss')
    ax2.legend()
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(im0_bin)
ax3.title.set_text('Binary Image 0')
ax3.plot(p_img0[0, :], p_img0[1, :], label="reconstruction")
ax3.legend()
ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(im1_bin)
ax4.title.set_text('Binary Image 1')
ax4.plot(p_img1[0, :], p_img1[1, :], label="reconstruction")
ax4.legend()
