import cv2
import math
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from .utils import find_longest_path

#%%

class EpipolarReconstructor:    
    
    r"""
       Reconstruction through Epipolar geometry. 

       Arguments
       ----------
       
       img_0, img_1 : array
           Photo in cam.0 and Photo in cam.1
       
       A0, A1, dist0, dist1, P0, P1, R, T : array
           camera matrix
           A0 and A1 are the intrinsic camera matrices for camera 0 and camera 1.
           dist0 and dist1 represent the radial distortion coefficients.
           P0 and P1 are the projection matrices for camera 0 and camera 1.
           R stands for the rotation matrix, and T represents the translation vector.
           
       bin_threshold : int
           Set pixel value greater than or equal to bin_threshold to 255 and less than bin_threshold to 0, The recommended value is 200.
       
       tip_estimator_params_0, tip_estimator_params_1 : store data in MATLAB
           Defines the initial points and initial directions of the skeleton in camera 0 and camera 1.

       Sources
       ----------
       Hartley, R., & Zisserman, A. (2004). Epipolar Geometry and the Fundamental Matrix. In Multiple View Geometry in Computer Vision (pp. 239-261). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511811685.014
    """
    
    def __init__(self, img_0, img_1, A0, A1, dist0, dist1, P0, P1, R, T, bin_threshold, tip_estimator_params_0, tip_estimator_params_1):
        # Initialise variables:
        self.img_0 = img_0 # Photo in cam.0
        self.img_1 = img_1 # Photo in cam.1
        
        # camera matrix
        self.A0 = A0 
        self.A1 = A1 
        self.dist0 = dist0
        self.dist1 = dist1
        self.P0 = P0 
        self.P1 = P1
        self.R = R
        self.T = T
        
        self.bin_threshold = bin_threshold
        
        self.tip_estimator_params_0 = tip_estimator_params_0
        self.tip_estimator_params_1 = tip_estimator_params_1
          
    def compute_fundamental_matrix(self, K1, K2, R, t):
        r"""
           By using fundamental matrix, the epipolar line in one camera can be generated from a point in anather camera. 
           The fundamental matrix can be calculated from camera matrix using the formula (9.2) in [1]. 
           
           Sources
           ----------
           [1] Hartley, R., & Zisserman, A. (2004). Epipolar Geometry and the Fundamental Matrix. In Multiple View Geometry in Computer Vision (pp. 239-261). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511811685.014

         """
        A = np.dot(K1, np.dot(R.T, t)).flatten()
        C = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
        ret = np.dot(np.linalg.inv(K2).T, np.dot(R, np.dot(K1.T, C)))
        return ret
        
    def get_2D (self, plot = False):
        
        r"""
           Obtain the skeletons of images from camera 1 and camera 2, and arrange the pixel coordinates of the skeletons in the order of the path from the starting point.

           Arguments
           ----------
           plot : boolean
               If True, images in camera 0 and camera 1 are displayed.
           
           data_cam_0, data_cam_1 : array in size (n,2), n is the number of pixels in the skeletons of images from camera 0 and camera 1.
               The first two elements of the return value of the 'get_2D' function.
               The 2D coordinates of each point in the skeletons of images from camera 0 and camera 1 are arranged in the order of the path from the starting point.           
           
           kan_img_open_0, kan_img_open_1 : array
               The third and fourth elements of the return value of the 'get_2D' function.
               Images in camera 0 and camera 1
         """
         
        
        # Pick the desired points for projection
        # img in cam.0
        if len(self.img_0.shape) == 3:
            img = cv2.cvtColor(self.img_0, cv2.COLOR_BGR2GRAY)
        else:
            img = self.img_0    

        param_dict = loadmat(self.tip_estimator_params_0)
        p_start = param_dict["p_start"]
        exit_dir = param_dict["exit_dir"]
        
        # filter noise
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        kan_img_open_0 = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)

        kan_img_open_0[kan_img_open_0 >= self.bin_threshold] = 255
        kan_img_open_0[kan_img_open_0 < self.bin_threshold] = 0

        # invert the image 
        kan_img_open_0 = 255-kan_img_open_0

        # get skeleton of img0
        img_skel = np.array(skeletonize(kan_img_open_0/255))*255
        m,n = np.where(img_skel==255)    
        skeleton_0 = np.array([n,m]).T
        
        # selekt and arrange the pixel coordinates of the skeleton_0
        sk_0_ordered = find_longest_path(skeleton_0, p_start)
        
        # Remove distortion so that the epipolar line will not shift
        sk_0_ordered = cv2.undistortPoints(sk_0_ordered.astype(np.float32), self.A0, self.dist0, P = self.A0)
        sk_0_ordered = np.squeeze(sk_0_ordered)
            
        if(plot):
            plt.figure(1)
            plt.scatter(sk_0_ordered[:,0],sk_0_ordered[:,1])
            plt.xlim(0, self.img_0.shape[1])
            plt.ylim(0, self.img_0.shape[0])
            plt.xlabel('u')
            plt.ylabel('v')
            plt.title('skeleton in cam.0')
            ax = plt.gca() 
            ax.invert_yaxis()
               
        # Positiondata for the skeleton in img0
        data_cam_0 = sk_0_ordered 
        
        # Read img in cam.1 and skeletonize    
        img_bgr = self.img_1
        
        if len(img_bgr.shape) == 3:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = img_bgr

        tip_estimator_params = self.tip_estimator_params_1
        param_dict = loadmat(tip_estimator_params)
        p_start = param_dict["p_start"]
        exit_dir = param_dict["exit_dir"]
        blank_idc = param_dict["blank_idc"]
        
        # filter noise
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        kan_img_open_1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)

        kan_img_open_1[kan_img_open_1 >= self.bin_threshold] = 255
        kan_img_open_1[kan_img_open_1 < self.bin_threshold] = 0

        # invert the image 
        kan_img_open_1 = 255-kan_img_open_1
        
        # get skeletion of img1
        img_skel = np.array(skeletonize(kan_img_open_1 /255))*255
        m,n = np.where(img_skel==255)
        skeleton_1 = np.array([n,m]).T
        
        # selekt and arrange the pixel coordinates of the skeleton_1
        sk_1_ordered = find_longest_path(skeleton_1, p_start)
        
        # Remove distortion, the identification of insertion point of epiplorline should be applied in undistorted image
        sk_1_ordered = cv2.undistortPoints(sk_1_ordered.astype(np.float32), self.A1, self.dist1, P = self.A1)
        sk_1_ordered = np.squeeze(sk_1_ordered)
            
        if(plot): 
            plt.figure(2)
            plt.scatter(sk_1_ordered[:,0],sk_1_ordered[:,1])
            plt.xlim(0, self.img_1.shape[1])
            plt.ylim(0, self.img_1.shape[0])
            plt.xlabel('u')
            plt.ylabel('v')
            plt.title('skeleton in cam.1')
            ax = plt.gca() 
            ax.invert_yaxis()
               
        # Positiondata for the skeleton in img0
        data_cam_1 = sk_1_ordered 
        
        return (data_cam_0, data_cam_1, kan_img_open_0, kan_img_open_1)
    
    def get_3D (self, data_cam_0=None, data_cam_1=None, interval = 20, plot=True):
        
        r"""
           Obtain the 3D coordinates of the skeleton based on the 2D coordinates data in camera 0 and camera 1.
           To establish correspondences for each point in the skeleton from camera 0 and camera 1, the points in the skeleton from camera 0 are projected into camera 1 using the fundamental matrix. 
           The intersection points between the skeleton in camera 1 and the projection lines are identified as the corresponding points from camera 0.
           
           Arguments
           ---------
           data_cam_0, data_cam_1 : array in size (n,2), n is the number of pixels in the skeletons of images from camera 0 and camera 1.
               The 2D coordinates of each point in the skeletons of images from camera 0 and camera 1, which arranged in the order of the path from the starting point.           
           
           interval : int
               Determines how often projection lines are cast from camera 0 in camera 1 to search for corresponding points in camera 2. 
               Increasing the interval value appropriately can reduce the algorithm's computation time.
               
           plot : boolean
               If True, images in camera 0 and camera 1 as well as the 3D reconstruction, are displayed.
         """
        
        # Extract rotation and translation matrices from projection matrices. Calculate fundamental matrix 

        R_t_0 = np.dot(np.linalg.inv(self.A0),self.P0)
        R_t_1 = np.dot(np.linalg.inv(self.A1),self.P1)

        R0 = R_t_0[:, :3] 
        t0 = R_t_0[:, 3] 
        t0 = t0.reshape(-1, 1) 

        R1 = R_t_1[:, :3] 
        t1 = R_t_1[:, 3]
        t1 = t1.reshape(-1, 1) 

        R1_0 = np.dot(R1, R0.T)
        t1_0 = np.dot(R1, -np.dot(R0.T, t0)) + t1

        F = self.compute_fundamental_matrix(self.A0, self.A1, R1_0, t1_0)
        F = np.array([[float(F[i][j]) for j in range(3)] for i in range(3)], dtype=np.float32)
        
        if data_cam_0.any()==None or data_cam_1.any()==None:
            data_cam_0, data_cam_1 = self.get_2D (plot)
            
        else:
            if (plot):
                plt.figure(1)
                plt.scatter(data_cam_0[:,0],data_cam_0[:,1])
                plt.xlim(0, self.img_0.shape[1])
                plt.ylim(0, self.img_0.shape[0])
                plt.xlabel('u')
                plt.ylabel('v')
                plt.title('skeleton in cam.0')
                ax = plt.gca() 
                ax.invert_yaxis()
                
                plt.figure(2)
                plt.scatter(data_cam_1[:,0],data_cam_1[:,1])
                plt.xlim(0, self.img_1.shape[1])
                plt.ylim(0, self.img_1.shape[0])
                plt.xlabel('u')
                plt.ylabel('v')
                plt.title('skeleton in cam.1')
                ax = plt.gca() 
                ax.invert_yaxis()
                                
        sk_0_ordered = data_cam_0 [::interval]
        sk_1_ordered = data_cam_1
        
        pos_list = []
        data_3d_list = []

        for i in range(sk_0_ordered.shape[0]):
                      
            points_camera_0 = sk_0_ordered[i]
            
            # Calculate epilines in camera 1 corresponding to points in camera A
            line_B = cv2.computeCorrespondEpilines(points_camera_0[np.newaxis, :], 0, F) 
            line_B = np.array(line_B).ravel()
            a, b, c = line_B
            
            if (plot):
                plt.figure(1)
                plt.scatter(sk_0_ordered[i,0],sk_0_ordered[i,1],color='r')
                plt.pause(0.1)
                
                plt.figure(2)
                x_line_B = np.arange(0,self.img_1.shape[1],1)
                y_line_B = -(a * x_line_B  + c) / b
                plt.plot(x_line_B ,y_line_B,color='r' ,alpha =0.4)
                plt.pause(0.1)
                     
            # Find intersection of epilines in camera 0
            distances = np.array([abs(a * x + b * y + c) / np.sqrt(a**2 + b**2) for x, y in sk_1_ordered])

            sorted_indices = np.argsort(distances)
            sorted_skeleton_1 = [sk_1_ordered[i] for i in sorted_indices]
            
            distance = cdist(sorted_skeleton_1[0][np.newaxis, :], sorted_skeleton_1[1:10])
            distance =  np.array(distance).ravel()
            
            param_dict = loadmat(self.tip_estimator_params_1)
            p_start = param_dict["p_start"]
            
            if i == 0:
                point_1_list = []
                pre_point_1 = p_start
                
                for k in range(len(sorted_skeleton_1)):
                    
                    if cdist(sorted_skeleton_1[k][np.newaxis, :], pre_point_1) < 100:
                        points_camera_1 = sorted_skeleton_1[k]                                          
                        pos_camera = cv2.triangulatePoints(self.P0, self.P1, points_camera_0, points_camera_1)
                        pos_camera = np.swapaxes(pos_camera, 0, 1)
                        pos_camera = cv2.convertPointsFromHomogeneous(pos_camera)
                        pos_camera = np.array(pos_camera).ravel()
                        
                        pos = self.R@pos_camera + self.T
                        
                        pos_old = [0,0,0]
                        length = np.linalg.norm(pos - pos_old)
                        
                        pos_old = pos
                        
                        pre_point_1 = points_camera_1
                        sorted_indices_pre = sorted_indices[k]
                        point_1_list.append(list (points_camera_1))
                        
                        if (plot):
                            plt.figure(2)
                            plt.scatter(points_camera_1[0],points_camera_1[1],c="g")
                        
                        break
            else: 
                
                for k in range(len(sorted_skeleton_1)):
                    
                    if sorted_indices_pre < 5:
                        direction_bb = (sk_1_ordered[sorted_indices_pre+5][0]-sk_1_ordered[sorted_indices_pre][0]), (sk_1_ordered[sorted_indices_pre+5][1]-sk_1_ordered[sorted_indices_pre][1])              
                    elif sk_1_ordered.shape[0] - sorted_indices_pre < 6:
                        direction_bb = (sk_1_ordered[sorted_indices_pre][0]-sk_1_ordered[sorted_indices_pre-5][0]), (sk_1_ordered[sorted_indices_pre][1]-sk_1_ordered[sorted_indices_pre-5][1])              
                    else:
                        direction_bb = (sk_1_ordered[sorted_indices_pre+5][0]-sk_1_ordered[sorted_indices_pre-5][0]), (sk_1_ordered[sorted_indices_pre+5][1]-sk_1_ordered[sorted_indices_pre-5][1])    
                    
                    magnitude = math.sqrt(direction_bb[0]**2 + direction_bb[1]**2)
                    direction_bb_unit = (direction_bb[0]/magnitude, direction_bb[1]/magnitude)
                    
                    direction_ziel = (sk_1_ordered[sorted_indices[k]][0]-pre_point_1[0]), (sk_1_ordered[sorted_indices[k]][1]-pre_point_1[1])
                    magnitude = math.sqrt(direction_ziel[0]**2 + direction_ziel[1]**2)
                    direction_ziel_unit = (direction_ziel[0]/magnitude, direction_ziel[1]/magnitude)
                                   
                    if cdist(sorted_skeleton_1[k][np.newaxis, :], pre_point_1[np.newaxis, :]) < 50 and np.dot(direction_bb_unit, direction_ziel_unit) > 0.8:
                        points_camera_1 = sorted_skeleton_1[k]
                        pre_point_1 = points_camera_1
                        sorted_indices_pre = sorted_indices[k]
                        point_1_list.append(list (points_camera_1))
                        
                        if (plot):
                            plt.figure(2)                            
                            plt.scatter(points_camera_1[0],points_camera_1[1],color='r')               
                            plt.pause(0.1)                   
                        
                        pos_camera = cv2.triangulatePoints(self.P0, self.P1, points_camera_0, points_camera_1)
                        pos_camera = np.swapaxes(pos_camera, 0, 1)
                        pos_camera = cv2.convertPointsFromHomogeneous(pos_camera)
                        pos_camera = np.array(pos_camera).ravel()
                        
                        pos = self.R@pos_camera + self.T
                        
                        length = length + np.linalg.norm(pos - pos_old)
                        data_3d = np.append(pos,length)  
                        
                        pos_list.append(pos)
                        data_3d_list.append(data_3d)
                        
                        pos_old = pos  
                
                        break    
                
        if(plot):
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')       
            pos_list = np.array(pos_list)
            ax.scatter(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], c='r', marker='o')
            ax.set_xlabel('X')        
            ax.set_ylabel('Y')        
            ax.set_zlabel('Z')          
            ax.set_box_aspect([1, 1, 1])
            
        return np.stack(data_3d_list, 0 )
    
    
    def get_3D_faster (self, data_cam_0=None, data_cam_1=None, interval = 20, segments = 10, plot=True):
        
        r"""
           Instead of using projection lines to establish correspondences for each point in the skeletons of images from camera 0 and camera 1, 
           the 'get_3D_faster' method divides 'skeleton_0' and 'skeleton_1' into multiple segments, 
           ensuring that the points within each segment in camera 0 and camera 1 correspond evenly to each other.
           
           Arguments
           ---------
           data_cam_0, data_cam_1 : array in size (n,2), n is the number of pixels in the skeletons of images from camera 0 and camera 1.
               The 2D coordinates of each point in the skeletons of images from camera 0 and camera 1, which arranged in the order of the path from the starting point.           
           
           interval : int
               Determines how often projection lines are cast from camera 0 in camera 1 to search for corresponding points in camera 2. 
               Increasing the interval value appropriately can reduce the algorithm's computation time.
           
           segments : int
               The number of segments into which the skeleton in camera 0 and camera 1 is divided.
               
           plot : boolean
               If True, images in camera 0 and camera 1 as well as the 3D reconstruction, are displayed.
         """
        
        # Extract rotation and translation matrices from projection matrices. Calculate fundamental matrix 

        R_t_0 = np.dot(np.linalg.inv(self.A0),self.P0)
        R_t_1 = np.dot(np.linalg.inv(self.A1),self.P1)

        R0 = R_t_0[:, :3] 
        t0 = R_t_0[:, 3] 
        t0 = t0.reshape(-1, 1) 

        R1 = R_t_1[:, :3] 
        t1 = R_t_1[:, 3]
        t1 = t1.reshape(-1, 1) 

        R1_0 = np.dot(R1, R0.T)
        t1_0 = np.dot(R1, -np.dot(R0.T, t0)) + t1

        F = self.compute_fundamental_matrix(self.A0, self.A1, R1_0, t1_0)
        F = np.array([[float(F[i][j]) for j in range(3)] for i in range(3)], dtype=np.float32)
        
        if data_cam_0.any()==None or data_cam_1.any()==None:
            data_cam_0, data_cam_1 = self.get_2D (plot)
            
        else:
            if (plot):
                plt.figure(1)
                plt.scatter(data_cam_0[:,0],data_cam_0[:,1])
                plt.xlim(0, self.img_0.shape[1])
                plt.ylim(0, self.img_0.shape[0])
                plt.xlabel('u')
                plt.ylabel('v')
                plt.title('skeleton in cam.0')
                ax = plt.gca() 
                ax.invert_yaxis()
                
                plt.figure(2)
                plt.scatter(data_cam_1[:,0],data_cam_1[:,1])
                plt.xlim(0, self.img_1.shape[1])
                plt.ylim(0, self.img_1.shape[0])
                plt.xlabel('u')
                plt.ylabel('v')
                plt.title('skeleton in cam.1')
                ax = plt.gca() 
                ax.invert_yaxis()
                
        sk_0_ordered = data_cam_0 [::interval]
        sk_1_ordered = data_cam_1
        
        pos_list = []
        data_3d_list = []       

        for i in range(sk_0_ordered.shape[0]):
                     
            points_camera_0 = sk_0_ordered[i]
            points_camera_0 = np.array(points_camera_0, dtype = np.float32)
            points_undist_0 = cv2.undistortPoints(points_camera_0, self.A0, self.dist0, P = self.A0)
            
            # Calculate epilines in camera 1 corresponding to points in camera A
            line_B = cv2.computeCorrespondEpilines(points_undist_0, 0, F) 
            line_B = np.array(line_B).ravel()
            a, b, c = line_B
            
            if (plot):
                plt.figure(1)
                plt.scatter(sk_0_ordered[i,0],sk_0_ordered[i,1],color='r')
                plt.pause(0.1)
                
                plt.figure(2)
                x_line_B = np.arange(0,self.img_1.shape[1],1)
                y_line_B = -(a * x_line_B  + c) / b
                plt.plot(x_line_B ,y_line_B,color='r' ,alpha =0.4)
                plt.pause(0.1)
                    
            # Find intersection of epilines in camera 0
            distances = np.array([abs(a * x + b * y + c) / np.sqrt(a**2 + b**2) for x, y in sk_1_ordered])

            sorted_indices = np.argsort(distances)
            sorted_skeleton_1 = [sk_1_ordered[i] for i in sorted_indices]
            
            distance = cdist(sorted_skeleton_1[0][np.newaxis, :], sorted_skeleton_1[1:10])
            distance =  np.array(distance).ravel()
            
            # only one crosspoint for one epiline
            if max (distance) < 20:
                
                points_camera_1 = sorted_skeleton_1[0]                      
                pos_camera = cv2.triangulatePoints(self.P0, self.P1, points_camera_0, points_camera_1)
                pos_camera = np.swapaxes(pos_camera, 0, 1)
                pos_camera = cv2.convertPointsFromHomogeneous(pos_camera)
                pos_camera = np.array(pos_camera).ravel()
                
                pos = self.R@pos_camera + self.T
                
                if i==0:
                    pos_old = [0,0,0]
                    length = np.linalg.norm(pos - pos_old)
                else: 
                    length = length + np.linalg.norm(pos - pos_old)
                                  
                data_3d = np.append(pos,length)
                
                pos_list.append(pos)
                data_3d_list.append(data_3d)
                
                pos_old = pos              
                
                if(plot):                
                    plt.figure(2)
                    plt.scatter(points_camera_1[0],points_camera_1[1],color='r') 
                    
            # more than one crosspoint for one epiline             
            else:
                
                cam0_seg_start = i               
                cam1_seg_start = min(sorted_indices[0:10])
                
                for j in range(segments):
                    
                    cam0_seg_end = i - 1 + int((sk_0_ordered.shape[0] - i)*(j+1)/segments)
                    points_camera_0 = sk_0_ordered[cam0_seg_end]
                    points_camera_0 = np.array(points_camera_0, dtype = np.float32)
                    points_undist_0 = cv2.undistortPoints(points_camera_0, self.A0, self.dist0, P = self.A0)
                    
                    # Calculate epilines in camera 1 corresponding to points in camera A

                    line_B = cv2.computeCorrespondEpilines(points_undist_0, 0, F) 
                    line_B = np.array(line_B).ravel()
                    a, b, c = line_B
                    
                    if (plot):
                        plt.figure(1)
                        plt.scatter(sk_0_ordered[cam0_seg_end, 0], sk_0_ordered[cam0_seg_end, 1],color='r')
                        plt.pause(0.1)
                        
                        plt.figure(2)
                        x_line_B = np.arange(0,self.img_1.shape[1],1)
                        y_line_B = -(a * x_line_B  + c) / b
                        plt.plot(x_line_B ,y_line_B,color='r' ,alpha =0.4)
                        plt.pause(0.1)
                        
                    # Find intersection of epilines in camera 0
                    distances = np.array([abs(a * x + b * y + c) / np.sqrt(a**2 + b**2) for x, y in sk_1_ordered])

                    sorted_indices = np.argsort(distances)
                    
                    list = []
                    for t in range(10):
                        
                        if sorted_indices[t] > cam1_seg_start:
                            list.append (sorted_indices[t])
                        
                    cam1_seg_end = min(list)
                    
                    Interval = (cam1_seg_end -  cam1_seg_start - 1)/(cam0_seg_end - cam0_seg_start - 1)
                    
                    for k in range (cam0_seg_end - cam0_seg_start):
                        
                        points_camera_0 = sk_0_ordered[cam0_seg_start + k]
                        points_camera_0 = np.array(points_camera_0, dtype = np.float32)
                        points_undist_0 = cv2.undistortPoints(points_camera_0, self.A0, self.dist0, P = self.A0)
                        
                        points_camera_1 = sk_1_ordered[round(cam1_seg_start + Interval*k)]
                        
                        pos_camera = cv2.triangulatePoints(self.P0, self.P1, points_camera_0, points_camera_1)
                        pos_camera = np.swapaxes(pos_camera, 0, 1)
                        pos_camera = cv2.convertPointsFromHomogeneous(pos_camera)
                        pos_camera = np.array(pos_camera).ravel()
                        pos = self.R@pos_camera + self.T
                        
                        length = length + np.linalg.norm(pos - pos_old)
                        data_3d = np.append(pos,length)
                        
                        pos_list.append(pos)
                        data_3d_list.append(data_3d)
                        
                        pos_old = pos
                        
                    cam0_seg_start = cam0_seg_end
                    cam1_seg_start = cam1_seg_end
                     
                break
         
        if(plot):
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            
            pos_list = np.array(pos_list)
            ax.scatter(pos_list[:, 0], pos_list[:, 1], pos_list[:, 2], c='r', marker='o')
            ax.set_xlabel('X')        
            ax.set_ylabel('Y ')        
            ax.set_zlabel('Z')
            
            ax.set_box_aspect([1, 1, 1])
            
        return np.stack(data_3d_list, 0 )
       

