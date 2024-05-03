from . import casadi, np, torch
import networkx as nx
from copy import copy
from sklearn.neighbors import BallTree

def brute_force_distance_norm(A, B, p):
    # Calculate distances using the p-norm
    distances = torch.cdist(A, B, p=p)**p
    # Find the index of the nearest point for each point in A
    return distances, torch.argmin(distances, dim=1)

def ball_tree_norm(A, B, p):
    # Convert tensors to numpy arrays
    A_np = A.detach().numpy()
    B_np = B.detach().numpy()
    # Create Ball Tree and query with A
    tree = BallTree(B_np, metric='minkowski', p=p)
    _, ind = tree.query(A_np, k=1)
    # Convert results back to tensor
    return torch.tensor(ind.ravel())

def find_longest_path(skeleton_pixels: np.array, initial_point: np.array):
    """
    An algorithm that find the longest possible in the skeleton of a binary image by first building a directed graph of the skeleton and then using the
    dag_longest_path function from networkx to find the longest path. The algorithm returns the longest path as an array of skeleton pixels.
    """
    # First build a directed graph of the skeleton
    G = nx.DiGraph()
    G.add_nodes_from(range(skeleton_pixels.shape[0]))
    has_edge = np.zeros((skeleton_pixels.shape[0], ), dtype=bool)
    
    # Now add the edges starting from the skeleton pixel closest to initial_point
    initial_index = np.argmin(np.linalg.norm(skeleton_pixels - initial_point, axis=1))
    current_index = copy(initial_index)
    has_edge[current_index] = True
    bifurcation_points = [[current_index]]

    # Now that the first path has been found iterate over the bifurcation points and add edges in the same way
    for bifurcation in bifurcation_points:
        current_index = int(bifurcation[0])
        new_bifurcation_points = set_edges_branch(skeleton_pixels, current_index, has_edge, G, initial_index)
        bifurcation_points.extend(new_bifurcation_points)
    # Now that the graph is built, find the longest path
    longest_path = nx.dag_longest_path(G)
    return skeleton_pixels[longest_path, :]


def set_edges_branch(skeleton_pixels: np.array, current_index: int, has_edge: np.array, G: nx.DiGraph, initial_index: int):
    """
    Checks one whole branch of the skeleton. By iteratively adding edges to the graph G by adding the next skeleton pixel to the graph. If there are multiple
    skeleton pixels that are connected to the current skeleton pixel, the remaining ones are added to a list of bifurcation points that still need to be
    checked. The algorithm returns a list of bifurcation points that still need to be checked.
    """
    bifurcation_points = []
    while True:
        # Find all the skeleton pixels that have an inf-norm distance to the current skeleton pixel of 1
        current_pixel = skeleton_pixels[current_index, :]
        next_pixels_bool = (np.linalg.norm(skeleton_pixels - current_pixel, axis=1, ord=np.inf) == 1) & (~has_edge)
        next_pixels_idx = np.argwhere(next_pixels_bool)
        if not all(next_pixels_idx.shape):
            break
        next_pixels = skeleton_pixels[next_pixels_idx, :]
        # Now add edges to all the next pixels that have not been added yet
        for i in range(next_pixels.shape[0]):
            if current_index == initial_index:
                # Ensure that the initial pixel is always contained in the path
                G.add_edge(int(current_index), int(next_pixels_idx[i]), weight=1e8)
                has_edge[next_pixels_idx[i]] = True
            else:
                G.add_edge(int(current_index), int(next_pixels_idx[i]))
                has_edge[next_pixels_idx[i]] = True

        # Check whether there are multiple skeleton pixels that are connected to the current skeleton pixel. If so, add them to the list of bifurcation points,
        # excluding current_index.
        if next_pixels_idx.shape[0] > 1:
            # Set the current index to the index of the pixel from the next_pixels that follows the previous trend of the skeleton the best. This is done by
            # computing the dot product of the vector from the current pixel to the next pixel with the vector from the previous 5 pixels to the current pixel.
            # The pixel with the highest dot product is the one that follows the previous trend the best.
            current_vectors = skeleton_pixels[current_index, :] - skeleton_pixels[current_index-5:current_index, :]
            current_vector = np.mean(current_vectors / np.linalg.norm(current_vectors, axis=1)[:, None], axis=0)/np.linalg.norm(np.mean(current_vectors, axis=0))
            next_vectors = skeleton_pixels[next_pixels_idx[:, 0], :] - skeleton_pixels[current_index, :]
            next_vectors = next_vectors / np.linalg.norm(next_vectors, axis=1)[:, None]
            current_index = int(next_pixels_idx[np.argmax(next_vectors @ current_vector.T), :])
            bifurcation_points.append(next_pixels_idx[next_pixels_idx[:, 0] != current_index, :])
        else:
            current_index = int(next_pixels_idx[0])
    return bifurcation_points


def discretizeODE(odefun, method, dim, dt, nTubes, s, x, dtype=casadi.MX):
    butcher = getMethod(method)
    a = butcher[0:-1, 1:]
    b = butcher[-1, 1:]
    c = butcher[0:-1, 0]
    
    nStage = len(c)
    k = dtype(dim, nStage)
    
    for iStage in range(nStage):
        k[:, iStage] = odefun(s + dt*c[iStage], x + dt * k @ a[iStage,:].reshape(nStage, 1))
        
    xplus = x + dt * k @ b.reshape(nStage,1)
    return xplus

def getMethod(method):
    if method == "rk1" or method == "euler":
        butcher = np.diag([0,1])
    
    elif method == "rk2" or method == "midpoint":
        butcher  = np.diag([0, 1/2, 1]) + np.diag([1/2, 0], -1)
        
    elif method == "heun":
        butcher  = np.diag([0, 1, 1/2]) + np.diag([1, 1/2], -1)
        
    elif method == "rk3" or method == "simpson":
        butcher = np.diag([0, 1/2, 2, 1/6]) + np.diag([1/2, -1, 2/3], -1) + np.diag([1, 1/6], -2)
        
    elif method == "rk4":
        butcher = np.diag([0, 1/2, 1/2, 1, 1/6])
        butcher[1:4, 0] = np.array([1, 1, 2])/2
        butcher[-1, 1:4] = np.array([1, 2, 2])/6
        
    elif method == "3/8":
        butcher = np.diag([0, 1/3, 1, 1, 1/8]) + np.diag([1/3, -1/3, -1, 3/8], -1) + \
            np.diag([2/3, 1, 3/8], -2) + np.diag([1, 1/8], -3)
        
    return butcher


def fromWorld2Img(pos, A, dist, P, R, T):
    
    if pos.dtype != torch.float32:
        pos = pos.float()
   
    R_t = torch.linalg.solve(A, P).float()
    
    # Convert from local coordinate system of the CTCR to the world coordinate system 
    pos_camera = torch.linalg.solve(R, pos-T.reshape(3,1))

    pos_camera_h = torch.concat((pos_camera, torch.ones((1, pos_camera.shape[1]))), 0)
    
    # Calculate normalized camera coordinates in Camera
    camPoint = torch.matmul(R_t, pos_camera_h)
    normCamPoint = camPoint[:2,:] / camPoint[2,:]

    # Calculate Distortion
    k1, k2, p1, p2, k3 = dist
    r = normCamPoint[0,:]**2 + normCamPoint[1,:]**2 

    radDist = 1.0 + k1*r + k2*r**2 + k3*r**3
    tangDistX = 2*p1*normCamPoint[0,:]*normCamPoint[1] + p2*(r + 2*normCamPoint[0,:]**2)
    tangDistY = p1*(r + 2*normCamPoint[1,:]**2) + 2*p2*normCamPoint[0,:]*normCamPoint[1,:]

    # Add distortion
    x = radDist*normCamPoint[0,:] + tangDistX
    y = radDist*normCamPoint[1,:] + tangDistY

    # Calculate pixel coordinates in Camera
    pixel =  torch.matmul(A,torch.stack([x, y, torch.ones_like(x)],0))[:2]
    
    return pixel


def fromWorld2ImgCasadi(pos, A, dist, P, R, T):
   
    R_t = np.linalg.solve(A, P)
    
    # Convert from local coordinate system of the CTCR to the world coordinate system 
    pos_camera = R.T@(pos-T.reshape(3,1)) # casadi.solve?

    pos_camera_h = casadi.vertcat(pos_camera, casadi.MX.ones(1, pos_camera.shape[1]))
    
    # Calculate normalized camera coordinates in Camera
    camPoint = R_t @ pos_camera_h
    normCamPoint = camPoint[:2,:] / casadi.repmat(camPoint[2,:],2,1)

    # Calculate Distortion
    k1, k2, p1, p2, k3 = dist
    r = normCamPoint[0,:]**2 + normCamPoint[1,:]**2 

    radDist = 1.0 + k1*r + k2*r**2 + k3*r**3
    tangDistX = 2*p1*normCamPoint[0,:]*normCamPoint[1,:] + p2*(r + 2*normCamPoint[0,:]**2)
    tangDistY = p1*(r + 2*normCamPoint[1,:]**2) + 2*p2*normCamPoint[0,:]*normCamPoint[1,:]

    # Add distortion
    x = radDist*normCamPoint[0,:] + tangDistX
    y = radDist*normCamPoint[1,:] + tangDistY

    # Calculate pixel coordinates in Camera
    pixel = (A @ casadi.vertcat(x, y, casadi.MX.ones(x.shape)))[:2,:]
    
    return pixel


def spaceCarving(images, cam_params, x_bounds, y_bounds, z_bounds, resolution=0.1):
    xgrid, ygrid, zgrid = torch.meshgrid(torch.arange(x_bounds[0], x_bounds[1]+resolution, resolution), torch.arange(y_bounds[0], y_bounds[1]+resolution, resolution), torch.arange(z_bounds[0], z_bounds[1]+resolution, resolution), indexing='ij')
    pts_3d = torch.stack([xgrid.flatten(), ygrid.flatten(), zgrid.flatten()], 1)
    voting = np.zeros((pts_3d.shape[0], len(images)), bool)
    for i in range(len(images)):
        proj_pts = fromWorld2Img(pts_3d.T, **cam_params[i]).round().T
        filter_out = (proj_pts[:,0] < 0) | (proj_pts[:,1] < 0) | (proj_pts[:,0] >= images[i].shape[1]) | (proj_pts[:,1] >= images[i].shape[0])
        pts_3d = pts_3d[~filter_out,:]
        proj_pts = proj_pts[~filter_out,:]
        voting = voting[~filter_out,:]
        voting[:,i] = images[i][proj_pts[:,1].int(), proj_pts[:,0].int()]
    return pts_3d[voting.all(1),:], xgrid, ygrid, zgrid
        
"""
    Alle Hilfsfunktionen
"""