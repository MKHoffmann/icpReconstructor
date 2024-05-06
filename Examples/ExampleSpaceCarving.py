import sys
import os

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from icpReconstructor.torch_reconstruction import image_to_idx, camera_folder_to_params, PixelDataset
from icpReconstructor.utils import spaceCarvingReconstruction, find_longest_path
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

#%% Image data
"""
    Load the images, binarize them and load the indices of the pixels that are part of the cannula.
"""

im0_bin = plt.imread("im0.png") > 0
im1_bin = plt.imread("im1.png") > 0
p0_img = image_to_idx(im0_bin)
p1_img = image_to_idx(im1_bin)

dataset = PixelDataset([p0_img, p1_img])

start = time()

images = [im0_bin, im1_bin]
x_bounds = [-0.01, 0.07]
y_bounds = [-0.04, 0.02]
z_bounds = [0, 0.16]

path, pts_3d = spaceCarvingReconstruction(images, cam_params, x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)

path_plot = path.detach().numpy()
s_path = np.zeros(path.shape[0])
for i in range(1, path.shape[0]):
    s_path[i] = s_path[i-1] + np.linalg.norm((path[i]-path[i-1]), 2)
    
print(f"The space carving took {time()-start} s.")
#%%
pts_3d_plot = pts_3d.detach().numpy()
fig = plt.figure()
ax1 = fig.add_subplot(projection="3d")
ax1.plot(pts_3d_plot[:, 0], pts_3d_plot[:, 1], pts_3d_plot[:, 2], '.', linewidth=1, label='Space Carving')
ax1.axis('equal')
ax1.set_xticklabels
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax1.legend(fontsize=20)

#%%
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(path_plot[:,0],path_plot[:,1],path_plot[:,2])
ax.axis('equal')