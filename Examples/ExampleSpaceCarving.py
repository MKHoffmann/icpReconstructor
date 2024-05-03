import sys
import os

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from icpReconstructor.torch_reconstruction import image_to_idx, camera_folder_to_params, CannulaPixelDataset
from icpReconstructor.utils import spaceCarving, find_longest_path
from skimage.morphology import skeletonize_3d
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

dataset = CannulaPixelDataset([p0_img, p1_img])

start = time()

initial_point = np.zeros((3,))
x_bounds = [-0.01, 0.07]
y_bounds = [-0.04, 0.02]
z_bounds = [0, 0.16]
resolution = 0.001

pts_3d, xgrid, ygrid, zgrid = spaceCarving([im0_bin, im1_bin], cam_params, x_bounds, y_bounds, z_bounds, resolution)

def get_binary_grid(pts_3d, x_bounds, y_bounds, z_bounds, resolution):
    # Calculate grid size
    grid_shape = (
        int((x_bounds[1] - x_bounds[0]) / resolution) + 1,
        int((y_bounds[1] - y_bounds[0]) / resolution) + 1,
        int((z_bounds[1] - z_bounds[0]) / resolution) + 1
    )
    
    # Initialize the binary grid
    binary_3d_grid = np.zeros(grid_shape, dtype=bool)
    
    # Convert pts_3d to grid indices
    grid_indices = ((pts_3d - torch.tensor([x_bounds[0], y_bounds[0], z_bounds[0]])) / resolution).round().int()
    
    # Update the binary grid
    for index in grid_indices:
        binary_3d_grid[index[0], index[1], index[2]] = True
    
    return binary_3d_grid

#%%
binary_3d_grid = get_binary_grid(pts_3d, x_bounds, y_bounds, z_bounds, resolution)
skeletonized_grid = skeletonize_3d(binary_3d_grid)

[xs, ys, zs] = np.where(skeletonized_grid)
dist_grid = ((xgrid-initial_point[0])**2+(ygrid-initial_point[1])**2+(zgrid-initial_point[2])**2).detach().numpy()
dist_min = np.min(dist_grid)
initial_point_grid = np.stack(np.where(dist_grid == dist_min), 1)

path_grid = find_longest_path(np.stack((xs, ys, zs), 1), initial_point_grid)
path = np.stack([xgrid[path_grid[:,0], path_grid[:,1], path_grid[:,2]], ygrid[path_grid[:,0], path_grid[:,1], path_grid[:,2]], zgrid[path_grid[:,0], path_grid[:,1], path_grid[:,2]]], 1)
path = np.concatenate([np.zeros((1,3)), path], 0)
s_path = np.zeros(path.shape[0])
for i in range(1, path.shape[0]):
    s_path[i] = s_path[i-1] + np.linalg.norm((path[i]-path[i-1]), 2)
    
print(f"The space carving took {time()-start} s.")
#%%
pts_3d_plot = pts_3d.detach().numpy()
fig = plt.figure()
ax1 = fig.add_subplot(projection="3d")
ax1.plot(pts_3d_plot[:, 0], pts_3d_plot[:, 1], pts_3d_plot[:, 2], '.', linewidth=1, label='Space Carving')
ax1.plot(xgrid[skeletonized_grid].detach().numpy(), ygrid[skeletonized_grid].detach().numpy(), zgrid[skeletonized_grid].detach().numpy(), '.', linewidth=1, label='Skeleton')
ax1.axis('equal')
ax1.set_xticklabels
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax1.legend(fontsize=20)

#%%
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(path[:,0],path[:,1],path[:,2])
ax.axis('equal')