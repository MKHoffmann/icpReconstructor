import casadi
import numpy as np
import torch
import networkx as nx
from copy import copy
from sklearn.neighbors import BallTree
from torch.utils.data import Dataset
from os.path import sep
from skimage.morphology import skeletonize_3d

def image_to_idx( img ):
    """
        Transform the binary image (img) to coordinates.
    """
    im_idc = np.where(img.T)
    return np.stack(im_idc, 1)


def camera_folder_to_params( camera_folder, cams, package="torch", device=torch.device('cpu') ):
    """
        Read the parameters from a folder using predefined file names.

        Arguments
        ----------
        camera_folder : str
            Folder containing the parameter files.

        cams : int or list
            Either the number of cameras to be used or a list of camera indices.
    """
    i_cams = cams if type(cams) is list else list(range(cams))

    param_list = []
    path = camera_folder if camera_folder[-1] == sep else camera_folder + sep
    load_and_tensor = {"torch":lambda x: torch.from_numpy(np.load(path + x)).float().to(device), "casadi":lambda x: np.load(path + x)}[package]
    for i in i_cams:
        param_list.append({
            "A": load_and_tensor(f"calib_A{i}.npy"),
            "dist": load_and_tensor(f"calib_dist{i}.npy"),
            "P": load_and_tensor(f"calib_P{i}_.npy"),
            "R_cam0_world": load_and_tensor("cs_conversion_R.npy"),
            "T_cam0_world": load_and_tensor("cs_conversion_T.npy")})
    return param_list


class PixelDataset(Dataset):
    """
        Dataset of all the pixel positions that correspond to the CR.

        Arguments
        ----------
            p_list : list
                List containing the pixel positions.

        Returns
        ----------
            p : m-by-2 tensor
                Tensor of pixel positions.

            img_idx_data : m-by-1 tensor
                Tensor containing the index of the corresponding camera.
    """

    def __init__( self, p_list, device=torch.device('cpu') ):
        self.p = None
        self.img_idx_data = None
        for i in range(len(p_list)):
            p = p_list[i]
            self.p = p if self.p is None else np.concatenate((self.p, p), 0)
            self.img_idx_data = i * np.ones((p.shape[0],), dtype=np.int8) if self.img_idx_data is None else np.concatenate(
                (self.img_idx_data, i * np.ones((p.shape[0],), dtype=np.int8)), 0)

    def __len__( self ):
        return self.p.shape[0]

    def __getitem__( self, idx ):
        return (self.p[idx, :], self.img_idx_data[idx])

def brute_force_distance_norm(A, B, p):
    # Calculate distances using the p-norm
    distances = torch.cdist(A, B, p=p)**p
    # Find the index of the nearest point for each point in A
    return distances, torch.argmin(distances, dim=1)

def ball_tree_norm(A, B, p):
    r"""
        Efficiently finds the nearest neighbors from one set of points in A to another set in B using the Minkowski distance metric
        parameterized by p. The function leverages a Ball Tree structure for fast nearest neighbor searches in high-dimensional spaces.
    
        The function converts PyTorch tensors to NumPy arrays, uses the BallTree implementation from sklearn for querying the nearest
        neighbor, and finally converts the results back to a PyTorch tensor.
    
        Arguments
        ----------
        A : torch.Tensor
            A tensor of points for which nearest neighbors in B need to be found. Each row corresponds to a different point.
    
        B : torch.Tensor
            A tensor of points constituting the search space for nearest neighbors. Each row corresponds to a different point.
    
        p : int
            The order of the Minkowski metric to be used for calculating distances. For example, p=2 corresponds to the Euclidean distance.
    
        Returns
        -------
        torch.Tensor
            A tensor containing the indices of the nearest neighbors in B for each point in A. The indices are 1-dimensional (flattened).
    
        Example
        -------
        Suppose you have two sets of points, `points_a` and `points_b`, represented as PyTorch tensors:
            points_a = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
            points_b = torch.tensor([[2.0, 1.0], [3.0, 2.0], [1.0, 2.0]])
    
        To find the nearest neighbor in `points_b` for each point in `points_a` using the Euclidean distance:
            nearest_indices = ball_tree_norm(points_a, points_b, p=2)
    
        This will return the indices of the closest points in `points_b` to each point in `points_a`.
    """

    # Convert tensors to numpy arrays
    A_np = A.cpu().detach().numpy()
    B_np = B.cpu().detach().numpy()
    # Create Ball Tree and query with A
    tree = BallTree(B_np, metric='minkowski', p=p)
    _, ind = tree.query(A_np, k=1)
    # Convert results back to tensor
    return torch.tensor(ind.ravel())

def find_longest_path(skeleton_pixels, initial_point):
    r"""
        Finds the longest path through a skeletonized representation of a binary image. This function constructs a directed graph from the 
        skeleton pixels, with the skeleton treated as a graph where junctions and end points become nodes, and connections between them 
        become directed edges. The function then calculates the longest path from a specified starting point to any node in the graph.
    
        The algorithm first identifies the closest skeleton pixel to an initial point and uses this as the starting node. It iteratively
        builds the graph by traversing the skeleton, adding edges from each node to its connected neighbors, while avoiding cycles. Finally,
        it utilizes the `dag_longest_path` method from `networkx` to determine the longest path in this directed acyclic graph (DAG).
    
        Arguments
        ----------
        skeleton_pixels : np.ndarray
            A 2D array where each row corresponds to the coordinates of a skeleton pixel in the image.
    
        initial_point : np.ndarray
            A 1D array representing the coordinates (x, y) of the initial point from which the nearest skeleton pixel is determined to start
            the pathfinding process.
    
        Returns
        -------
        np.ndarray
            An array of skeleton pixels representing the longest path found in the skeleton. Each row in the array is a pixel's coordinates
            along this path.
    
        Example
        -------
        Given a skeleton represented by an array of pixels `skeleton` and an initial point `init_point`:
            skeleton = np.array([[10, 10], [10, 11], [10, 12], [11, 12], [12, 12]])
            init_point = np.array([10, 9])
    
        Finding the longest path:
            longest_path = find_longest_path(skeleton, init_point)
    
        This would analyze the given skeleton structure, build the corresponding graph, and return the longest path starting from the
        point in the skeleton closest to [10, 9].
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


def discretizeODE(odefun, method, dim, dt, s, x, dtype=casadi.MX):
    r"""
        Discretizes an ordinary differential equation (ODE) using a specified numerical integration method. This function is designed
        to step through an ODE based on the initial conditions and parameters provided.
    
        This function primarily uses the Runge-Kutta method (or similar methods), parameterized by Butcher tableau coefficients to advance
        the state of the system over a single time step. It computes the state of the system at the next time step by evaluating several
        intermediate stages, each dependent on the results of the previous stages.
    
        .. note::
            The function requires the Butcher tableau for the numerical method, which is an array detailing the coefficients used in the
            numerical solution of ODEs. It supports any explicit Runge-Kutta method specified by its Butcher tableau.
    
        Arguments
        ----------
        odefun : callable
            The function defining the ODE. It should take at least two arguments: time `s` and state `x`, and return the derivative of the state.
    
        method : str
            The name of the integration method to use, which dictates the Butcher tableau. Integration method from the following list:
    
                - "rk1"/"euler"
                - "rk2"/"midpoint"
                - "heun"
                - "rk3"/"simpson"
                - "rk4"
                - "3/8"
                
        dim : int
            The dimension of the state vector `x`.
    
        dt : float
            The time step to advance the solution.
    
        s : float
            The current time.
    
        x : array_like
            The current state vector of the system.
    
        dtype : data type, optional
            The data type of the matrix that will hold the intermediate derivatives, by default `casadi.MX`, 
            which is specific to CasADi's symbolic framework.
    
        Returns
        -------
        xplus : array_like
            The state of the system at the next time step.
    
        Example
        -------
        Define an ODE function, such as a simple linear system:
            def linear_system(t, x):
                A = np.array([[0, 1], [-1, 0]])
                return A @ x
    
        Using the RK4 method:
            x_next = discretizeODE(linear_system, 'rk4', 2, 0.01, 1, 0, np.array([1, 0]))
    
        This would advance the system state from `np.array([1, 0])` at time `0` by `0.01` units using the RK4 integration method.
    """

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
    r"""
        Retrieves the Butcher tableau for a specified numerical integration method used in the solution of ordinary differential equations (ODEs).
        The Butcher tableau is a matrix representation that includes all coefficients necessary for the Runge-Kutta methods or its variants.
    
        The function supports a variety of integration methods, each associated with specific coefficients that dictate the steps of the
        integration process. These methods include simple Euler (first-order Runge-Kutta), Heun's method, Midpoint method (second-order Runge-Kutta),
        classical fourth-order Runge-Kutta, and others like the three-eighths rule (another fourth-order method).
    
        .. note::
            Each method is predefined with its respective coefficients arranged in a matrix format where the first row typically includes 
            the time step coefficients (c), and the last row includes the weights for the final summation (b). Intermediate rows provide
            the coefficients for the intermediate stages of the integration (a).
    
        Arguments
        ----------
        method : str
            The name of the integration method. Integration method from the following list:
    
                - "rk1"/"euler"
                - "rk2"/"midpoint"
                - "heun"
                - "rk3"/"simpson"
                - "rk4"
                - "3/8"
    
        Returns
        -------
        butcher : numpy.ndarray
            The Butcher tableau as a NumPy array. This matrix contains all coefficients needed to implement the specified numerical integration method.
    
        Example
        -------
        Retrieve the Butcher tableau for the classical fourth-order Runge-Kutta method:
            rk4_tableau = getMethod("rk4")
    
        This can be used to further implement the Runge-Kutta method for solving ODEs in a custom ODE solver.
    """

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


def fromWorld2Img(pos, A, dist, P, R_cam0_world, T_cam0_world):
    
    r"""
        Transforms 3D world coordinates into 2D pixel coordinates on an image plane using camera calibration parameters. This function
        applies several transformations including a change from world coordinates to camera coordinates, normalization, and distortion
        correction before finally using camera intrinsic parameters to map these points to pixel coordinates.
    
        The function ensures that all computations respect the data type required by PyTorch operations and handles both radial and
        tangential distortions as part of the transformation process. The final output is the pixel coordinates that correspond to
        the input 3D points as they would be captured by the camera.
    
        .. note::
            This function is particularly useful in computer vision and robotics for tasks like object tracking, 3D reconstruction,
            and augmented reality where accurate projection from 3D to 2D is crucial.
    
        Arguments
        ----------
        pos : torch.Tensor
            A 3xN matrix of 3D world coordinates, where N is the number of points.
    
        A : torch.Tensor
            The camera intrinsic matrix (3x3) including focal lengths and principal point.
    
        dist : tuple of floats
            The distortion coefficients (k1, k2, p1, p2, k3) for radial and tangential distortion.
    
        P : torch.Tensor
            The projection matrix (3x4) used to project 3D camera coordinates onto the image plane.
    
        R_cam0_world : torch.Tensor
            The rotation matrix (3x3) describing the orientation of the first camera in world coordinates.
    
        T_cam0_world : torch.Tensor
            The translation vector (3x1) describing the position of the first camera in world coordinates.
    
        Returns
        -------
        pixel : torch.Tensor
            A 2xN matrix where each column represents the pixel coordinates of the corresponding 3D point in the image.
    
        Example
        -------
        Define world points, camera intrinsic matrix `A`, distortion coefficients, projection matrix `P`, rotation matrix `R`,
        and translation vector `T`:
            pos = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
            A = torch.eye(3)
            dist = (0.1, 0.01, 0.001, 0.001, 0.0001)
            P = torch.eye(3)
            R = torch.eye(3)
            T = torch.tensor([0.1, 0.2, 0.3])
    
        Get pixel coordinates:
            pixels = fromWorld2Img(pos, A, dist, P, R, T)
    """

    if pos.dtype != torch.float32:
        pos = pos.float()
   
    R_t = torch.linalg.solve(A, P).float()
    
    # Convert from local coordinate system of the CR to the world coordinate system 
    pos_camera = R_cam0_world.T@(pos-T_cam0_world.reshape(3,1))
    pos_camera_h = torch.concat((pos_camera, torch.ones((1, pos_camera.shape[1]))), 0)
    
    # Calculate normalized camera coordinates in Camera
    camPoint = torch.matmul(R_t, pos_camera_h)
    normCamPoint = camPoint[:2,:] / camPoint[2,:]

    # Distort image
    k1, k2, p1, p2, k3 = dist
    r_sq = normCamPoint[0,:]**2 + normCamPoint[1,:]**2 

    radDist = 1.0 + k1*r_sq + k2*r_sq**2 + k3*r_sq**3
    tangDistX = 2*p1*normCamPoint[0,:]*normCamPoint[1] + p2*(r_sq + 2*normCamPoint[0,:]**2)
    tangDistY = p1*(r_sq + 2*normCamPoint[1,:]**2) + 2*p2*normCamPoint[0,:]*normCamPoint[1,:]

    # Add distortion
    x = radDist*normCamPoint[0,:] + tangDistX
    y = radDist*normCamPoint[1,:] + tangDistY

    # Calculate pixel coordinates in Camera
    pixel = torch.matmul(A, torch.stack([x, y, torch.ones_like(x)],0))[:2]
    
    return pixel


def fromWorld2ImgCasadi(pos, A, dist, P, R_cam0_world, T_cam0_world):
    r"""
        Transforms 3D world coordinates into 2D pixel coordinates using camera calibration parameters and the CasADi library for
        numerical computations. This function is designed for applications in optimization and control where gradient computations are
        necessary. It performs coordinate transformations, normalization, distortion correction, and final projection using camera
        intrinsic parameters.
    
        This function leverages CasADi's symbolic capabilities to handle operations, making it suitable for scenarios where the
        derivatives of the transformation process are needed for further optimizations.
    
        .. note::
            The use of CasADi's `MX` data type and operations ensures that the function is compatible with CasADi's automatic differentiation,
            which is crucial for gradient-based optimization tasks in computer vision and robotics.
    
        Arguments
        ----------
        pos : casadi.MX
            A 3xN matrix of 3D world coordinates, where N is the number of points. Should be of type casadi.MX for compatibility.
    
        A : numpy.ndarray
            The camera intrinsic matrix (3x3) including focal lengths and principal point. This is a static matrix.
    
        dist : tuple of floats
            The distortion coefficients (k1, k2, p1, p2, k3) used to model radial and tangential distortions.
    
        P : numpy.ndarray
            The projection matrix (3x3) used to project 3D camera coordinates onto the image plane. This is a static matrix.
    
        R_cam0_world : numpy.ndarray
            The rotation matrix (3x3) describing the orientation of the first camera in world coordinates.
    
        T_cam0_world : numpy.ndarray
            The translation vector (3x1) describing the position of the first camera in world coordinates.
    
        Returns
        -------
        pixel : casadi.MX
            A 2xN matrix where each column represents the pixel coordinates of the corresponding 3D point in the image. The output is
            of type casadi.MX to facilitate further symbolic operations if necessary.
    
        Example
        -------
        Define world points, camera intrinsic matrix `A`, distortion coefficients, projection matrix `P`, rotation matrix `R`,
        and translation vector `T`:
            pos = casadi.MX([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            A = np.eye(3)
            dist = (0.1, 0.01, 0.001, 0.001, 0.0001)
            P = np.eye(3)
            R = np.eye(3)
            T = np.array([0.1, 0.2, 0.3])
    
        Get pixel coordinates in a CasADi-compatible format:
            pixels = fromWorld2ImgCasadi(pos, A, dist, P, R, T)
    """

    R_t = np.linalg.solve(A, P)
    
    # Convert from local coordinate system of the CR to the world coordinate system 
    pos_camera = R_cam0_world.T@(pos-T_cam0_world.reshape(3,1))

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


def spaceCarving(images, cam_params, x_bounds, y_bounds, z_bounds, resolution=0.001):
    r"""
        Performs space carving to reconstruct a 3D volume from multiple images based on visibility from different camera angles.
        This method incrementally carves away the volume by projecting grid points into each image and checking if they are visible
        in all images. Points that are visible in all images are retained, while others are discarded.
    
        The function uses a grid of points defined within specified bounds and checks these points against each provided image
        using camera parameters to project 3D points into 2D image coordinates. Points that project outside the image bounds or
        into areas that do not match the image data are considered occluded and are removed from consideration.
    
        Arguments
        ----------
        images : list of binary images
            A list of images (2D arrays or tensors) from different views.
    
        cam_params : list of dicts
            A list of dictionaries, where each dictionary contains the camera parameters needed by the `fromWorld2Img` function
            to project 3D points onto each corresponding image. The dictionaries contain A, dist, p, R, and T.
    
        x_bounds : tuple of float
            The lower and upper bounds in the x-axis within which the space carving is performed.
    
        y_bounds : tuple of float
            The lower and upper bounds in the y-axis within which the space carving is performed.
    
        z_bounds : tuple of float
            The lower and upper bounds in the z-axis within which the space carving is performed.
    
        resolution : float, optional
            The resolution of the grid in all dimensions. Default is 0.001 units.
    
        Returns
        -------
        surviving_points : torch.Tensor
            A tensor of 3D points that are visible in all images, representing the carved space.
    
        xgrid, ygrid, zgrid : torch.Tensor
            The grid tensors in the x, y, and z dimensions respectively, which can be used for further analysis or visualization.
    
        Example
        -------
        Suppose you have multiple images and corresponding camera parameters:
            images = [img1, img2, img3]  # list of PyTorch tensors
            cam_params = [{'A': A1, 'dist': dist1, 'P': P1, 'R': R1, 'T': T1},
                          {'A': A2, 'dist': dist2, 'P': P2, 'R': R2, 'T': T2},
                          {'A': A3, 'dist': dist3, 'P': P3, 'R': R3, 'T': T3}]
    
        Define the bounds and resolution:
            x_bounds = (-1, 1)
            y_bounds = (-1, 1)
            z_bounds = (-1, 1)
            resolution = 0.01
    
        Perform space carving:
            carved_points, xg, yg, zg = spaceCarving(images, cam_params, x_bounds, y_bounds, z_bounds, resolution)
    """

    xgrid, ygrid, zgrid = torch.meshgrid(torch.arange(x_bounds[0], x_bounds[1]+resolution, resolution), torch.arange(y_bounds[0], y_bounds[1]+resolution, resolution), torch.arange(z_bounds[0], z_bounds[1]+resolution, resolution), indexing='ij')
    grid_points = torch.stack([xgrid.flatten(), ygrid.flatten(), zgrid.flatten()], 1)
    voting = np.zeros((grid_points.shape[0], len(images)), bool)
    for i in range(len(images)):
        proj_pts = fromWorld2Img(grid_points.T, **cam_params[i]).round().T
        filter_out = (proj_pts[:,0] < 0) | (proj_pts[:,1] < 0) | (proj_pts[:,0] >= images[i].shape[1]) | (proj_pts[:,1] >= images[i].shape[0])
        grid_points = grid_points[~filter_out,:]
        proj_pts = proj_pts[~filter_out,:]
        voting = voting[~filter_out,:]
        voting[:,i] = images[i][proj_pts[:,1].int(), proj_pts[:,0].int()]
    return grid_points[voting.all(1),:], xgrid, ygrid, zgrid


def spaceCarvingReconstruction(images, cam_params, p0=np.zeros((3,)), x_bounds=[-0.1, 0.1], y_bounds=[-0.1, 0.1], z_bounds=[0, 0.2], resolution=0.001):
    r"""
        Uses the spaceCarving method to generate a grid of 3D points potentially corresponding to the CR. The resulting grid is reduced to a skeleton that is
        sorted using the find_longest_path method. Depending on the camera setup this is nevcessary to deal with ambiguous data, as it can happen with two images.
    
        Arguments
        ----------
        images : list of binary images
            A list of 2D image tensors from different views, used to perform space carving.
    
        cam_params : list of dicts
            A list of dictionaries where each dictionary contains camera parameters needed for projecting 3D points into the images.
    
        p0 : np.ndarray, optional
            The origin point in 3D space from which to measure distances for pathfinding. Default is the zero vector (center of the bounds).
    
        x_bounds, y_bounds, z_bounds : list of float
            The bounds in the x, y, and z dimensions respectively, defining the region within which the carving and reconstruction are performed.
    
        resolution : float, optional
            The resolution of the grid in all dimensions, affecting the granularity of the carved space and the subsequent binary grid.
            Default is 0.001 units.
    
        Returns
        -------
        tuple
            Returns a tuple containing two elements:
            path : torch.Tensor
                A tensor containing the coordinates of the longest path found within the skeletonized grid. Useful for structural analysis.
            pts_3d : torch.Tensor
                The original point cloud derived from the space carving process before any skeletonization.
    
        Example
        -------
        Define a set of images and corresponding camera parameters:
            images = [torch.rand(100, 100) for _ in range(5)]
            cam_params = [{'A': torch.eye(3), 'dist': (0.1, 0.01, 0.001, 0.001, 0.0001), 'P': torch.eye(3), 'R': torch.eye(3), 'T': torch.tensor([0, 0, 0])} for _ in range(5)]
    
        Perform 3D reconstruction and analyze the structural properties:
            path, pts_3d = spaceCarvingReconstruction(images, cam_params)
    
        This would compute the 3D structure, skeletonize it, and determine the longest path for analysis.
    """

    pts_3d, xgrid, ygrid, zgrid = spaceCarving(images, cam_params, x_bounds, y_bounds, z_bounds, resolution)

    def get_binary_grid(pts_3d, x_bounds, y_bounds, z_bounds, resolution):
        # Calculate grid size
        grid_shape = (
            int((x_bounds[1] - x_bounds[0]) / resolution) + 1,
            int((y_bounds[1] - y_bounds[0]) / resolution) + 1,
            int((z_bounds[1] - z_bounds[0]) / resolution) + 1
        )
        
        # Initialize the binary grid
        binary_3d_grid = torch.zeros(grid_shape, dtype=bool)
        
        # Convert pts_3d to grid indices
        grid_indices = ((pts_3d - torch.tensor([x_bounds[0], y_bounds[0], z_bounds[0]])) / resolution).round().int()
        
        # Update the binary grid
        for index in grid_indices:
            binary_3d_grid[index[0], index[1], index[2]] = True
        
        return binary_3d_grid

    binary_3d_grid = get_binary_grid(pts_3d, x_bounds, y_bounds, z_bounds, resolution)
    skeletonized_grid = torch.from_numpy(skeletonize_3d(binary_3d_grid))

    [xs, ys, zs] = torch.where(skeletonized_grid)
    dist_grid = ((xgrid-p0[0])**2+(ygrid-p0[1])**2+(zgrid-p0[2])**2)
    dist_min = torch.min(dist_grid)
    initial_point_grid = torch.stack(torch.where(dist_grid == dist_min), 1)

    path_grid = torch.from_numpy(find_longest_path(torch.stack((xs, ys, zs), 1).detach().numpy(), initial_point_grid.detach().numpy()))
    path = torch.stack([xgrid[path_grid[:,0], path_grid[:,1], path_grid[:,2]], ygrid[path_grid[:,0], path_grid[:,1], path_grid[:,2]], zgrid[path_grid[:,0], path_grid[:,1], path_grid[:,2]]], 1)
    path = torch.concatenate([torch.zeros((1,3)), path], 0)
    
    return path, pts_3d
    

def generate_circle(n: int) -> np.ndarray:
    """
    Generate n points lying on the unit circle.

    Arguments
    ----------
        n (int): The number of points to generate.

    Returns
    -------
    numpy.array
        An n x 2 matrix where each row is a pair (cx, cy) lying on the unit circle.
    """
    
    # Generate n angles evenly spaced between 0 and 2pi
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Compute cx and cy for each angle
    cx = np.cos(angles)
    cy = np.sin(angles)

    # Combine cx and cy into a single matrix
    c_val = np.vstack((cx, cy)).T

    return c_val


"""
    Alle Hilfsfunktionen
"""