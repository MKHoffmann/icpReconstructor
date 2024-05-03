import sys

sys.path.append("..")

from ctcr_reconstructor.torch_reconstruction import TorchRotationFrame, image_to_idx, camera_folder_to_params, CannulaPixelDataset
from ctcr_reconstructor.casadi_reconstruction import CasadiCurveEstimator, Polynomial3Casadi, CasadiRotationFrame
from skimage.morphology import disk, binary_dilation
from ctcr_reconstructor.utils import fromWorld2Img
import torch
import casadi
import matplotlib.pyplot as plt
import PyCTR
import numpy as np
import reconstruction_by_projection
from time import time
from random import sample
from scipy.spatial.distance import cdist

#%% Initialization
l = torch.tensor([0.0750, 0.1300, 0.1900])  # length of the segments

"""
Load camera calibration files and simulate one set of cannulas.
"""
camera_folder = "camera_calibration_files"

cam_params = camera_folder_to_params(camera_folder, 2)
cam_params_cas = camera_folder_to_params(camera_folder, 2, package="casadi")

A0 = cam_params[0]["A"]
A1 = cam_params[1]["A"]
dist0 = cam_params[0]["dist"] # distortion parameters
dist1 = cam_params[1]["dist"]
P0 = cam_params[0]["P"]
P1 = cam_params[1]["P"]

R = cam_params[0]["R"]
T = cam_params[0]["T"]

plot_curvature = True
n_iter = 1

#%% Image data
"""
    Load the images, binarize them and load the indices of the pixels that are part of the cannula.
"""

im0_bin = plt.imread("im0.png") > 0
im1_bin = plt.imread("im1.png") > 0
p0_img = image_to_idx(im0_bin)
p1_img = image_to_idx(im1_bin)

dataset = CannulaPixelDataset([p0_img, p1_img])

#%% epipolar line matching
now = time()

tip_estimator_params_0 = camera_folder + "\\param_cam_0.mat"
tip_estimator_params_1 = camera_folder + "\\param_cam_1.mat"

bin_threshold = 200

reconstructor = reconstruction_by_projection.ReC(np.uint8((1-im0_bin[:,:,None])*255*np.ones((1,1,3))), np.uint8((1-im1_bin[:,:,None])*255*np.ones((1,1,3))), A0.detach().numpy(), A1.detach().numpy(), dist0.detach().numpy(), dist1.detach().numpy(), P0.detach().numpy(), P1.detach().numpy(), R.detach().numpy(), T.detach().numpy(), bin_threshold, tip_estimator_params_0, tip_estimator_params_1)

Data_cam_0, Data_cam_1, _, _ = reconstructor.get_2D(start_cam0 = 10,  end_cam0 = 10, start_cam1 = 10, end_cam1 = 10, plot=False)
Data_3d = reconstructor.get_3D(data_cam_0 = Data_cam_0, data_cam_1 = Data_cam_1, interval = 10, plot=False) #x, y, z, s

print(f"3D reconstruction took {time()-now} s.")

#%% Casadi
"""
    Initialize the curvature estimator by fitting the curvature function to the 
"""
curvature_options = {"continuous":False, "random_init":False, "end_no_curvature":False}

frac = 2
dataset_idc = sample(range(len(dataset)), int(len(dataset)/frac))

n = len(l)
start = time()
model = CasadiRotationFrame(l.detach().numpy(), n_steps=40)
ux = Polynomial3Casadi(model.opt, n, **curvature_options)
uy = Polynomial3Casadi(model.opt, n, **curvature_options)
model.ux = ux
model.uy = uy

model.add_u_constraint("x", -40, 40)
model.add_u_constraint("y", -40, 40)

cce = CasadiCurveEstimator(model, cam_params_cas, l[-1].detach().numpy(), dist_norm=2)

(ux, uy, uz) = cce.solve_3d(Data_3d[:,3], Data_3d[:,:3].T, lam_2=0)

frac = 2
loss_hist = []
for j in range(n_iter):
    dataset_idc = sample(range(len(dataset)), int(len(dataset)/frac))
    
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

curvature = TorchRotationFrame(l, integrator="dopri5", rotation_method="rotm", ux=ux, uy=uy, uz=uz)

s = torch.linspace(0, l[-1], 1000)
p_out = curvature.integrate(s).detach().numpy()

#%%
p_img0 = fromWorld2Img(torch.from_numpy(p_out[:, :3].T), A0, dist0, P0, R, T).detach().numpy()
p_img1 = fromWorld2Img(torch.from_numpy(p_out[:, :3].T), A1, dist1, P1, R, T).detach().numpy()

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
