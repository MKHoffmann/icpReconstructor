from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from PIL import Image
from skimage.morphology import (
    binary_opening, 
    binary_closing, 
    disk)

# NOTE: We explicitly guard the import of OpenCV as it is not a 
#       requirement for the entire library. However, it is required for 
#       this example.
try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required for this example. "
        "Install it using `pip install opencv-python`.") from e

from icpReconstructor.utils import (
    image_to_idx,
    fromWorld2Img)


def load_calibration_parameters(
        calibration_directory: Path,
        number_of_cameras: int) -> list[dict[str, torch.Tensor]]:
    """
        Load camera calibration parameters for each camera from the 
        `calibration_directory`. 

        Arguments
        ---------
        calibration_directory : Path
            Path to the directory containing the calibration parameters.
        number_of_cameras : int
            Number of cameras in the system.
        
        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of dictionaries containing the camera calibration 
            parameters for each camera. The dictionary contains the 
            following keys:
                - A : Camera matrix
                - dist : Distortion coefficients
                - P : Projection matrix
                - R : Rotation matrix
                - T : Translation vector
    """

    # Create empty array with camera calibration parameters
    calibration_parameters = [] * number_of_cameras

    for camera_index in range(number_of_cameras):
        # Camera matrix
        camera_matrix = torch.from_numpy(numpy.load(str(
            calibration_directory / 
            f"C{camera_index + 1}_camera_matrix.npy"))).to(
                torch.get_default_device()).float()
        
        # Distortion coefficients
        distortion_coefficients = torch.from_numpy(numpy.load(str(
            calibration_directory
            / f"C{camera_index + 1}_distortion_coefficients.npy"))).to(
                torch.get_default_device()).float()

        # Projection matrix
        projection_matrix = torch.from_numpy(numpy.load(str(
            calibration_directory
            / f"C{camera_index + 1}_projection_matrix.npy"))).to(
                torch.get_default_device()).float()
    
        # Stereo camera system
        # NOTE: We do not explicitly load the stereo camera system here
        #       as it is already handled through the individual camera
        #       projection matrices. However, R and T can be used in the
        #       fromWorld2Img function to move the camera according to
        #       the global coordinate system.
        R = torch.eye(3)
        T = torch.zeros(3)

        # Store parameters for camera
        parameters = {
            "A": camera_matrix,
            "dist": distortion_coefficients,
            "P": projection_matrix,

            "R_cam0_world": R,
            "T_cam0_world": T
        }

        calibration_parameters.append(parameters)

    return calibration_parameters


def load_images(
        filenames: list[str]): # BUG: -> list[cv2.typing.MatLike]:
    """
        Load images from the given `filenames`.

        Arguments
        ---------
        filenames : list[str]
            List of filenames to load images from.
        
        Returns
        -------
        list[cv2.typing.MatLike]
            List of images loaded from the given `filenames`.
    """
    
    # Initialize empty array to hold target images
    raw_images = []

    for i, filename in enumerate(filenames):
        bgr_image = cv2.imread(filename)

        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Convert image data from 0 - 255 to 0 - 1
        rgb_image = rgb_image / 255.0       

        # Store
        raw_images.append(rgb_image)
    
    return raw_images


def preprocess_images(
        images, #BUG: list[cv2.typing.MatLike],
        threshold_minimum_r: float = 200.0 / 255.0,
        threshold_maximum_g: float = 190.0 / 255.0,
        threshold_maximum_b: float = 140.0 / 255.0
    ) -> tuple[list[numpy.ndarray], list[torch.Tensor]]:
    """
        Preprocess the given `images` by applying a mask to the images 
        and binarizing them.

        Arguments
        ---------
        images : list[cv2.typing.MatLike]
            List of images to preprocess.
        threshold_minimum_r : float, optional
            Minimum threshold for the red channel, by default 200/255
        threshold_maximum_g : float, optional
            Maximum threshold for the green channel, by default 190/255
        threshold_maximum_b : float, optional
            Maximum threshold for the blue channel, by default 140/255
        
        Returns
        -------
        tuple[list[numpy.ndarray], list[numpy.ndarray]]]
            Tuple containing the preprocessed images and the image 
            indices.
    """
    
    # Initialize empty array to hold target images
    target_images = []
    target_image_indices = []

    for rgb_image in images:
        # Mask
        closing_mask = \
            (rgb_image[..., 0] > threshold_minimum_r) & \
            (rgb_image[..., 1] < threshold_maximum_g) & \
            (rgb_image[..., 2] < threshold_maximum_b)
        
        # Binarize the image
        masked_image = numpy.where(closing_mask, 1.0, 0.0)

        # Close pepper noise with radius of 10
        binary_closing(
            masked_image, 
            footprint=disk(radius=12),
            out=masked_image)
        binary_opening(
            masked_image, 
            footprint=disk(radius=4), 
            out=masked_image)

        # Convert to index
        image_indices = image_to_idx(numpy.asarray(masked_image))

        # Store
        target_images.append(masked_image)
        target_image_indices.append(image_indices)
    
    return target_images, target_image_indices


def compute_projected_points(
        points, 
        calibration_parameters) -> torch.Tensor:
    """
        Compute the projected points for the given `points` using the
        `calibration_parameters`.

        Arguments
        ---------
        points : torch.Tensor with shape (N, 3)
            Points to project.
        calibration_parameters : list[dict[str, torch.Tensor]]
            List of dictionaries containing the camera calibration 
            parameters for each camera.
        
        Returns
        -------
        torch.Tensor with shape (2, N, 2)
            Projected points for the given `points`.
    """

    number_of_cameras = len(calibration_parameters)

    projected_points = torch.empty((number_of_cameras, points.shape[0], 2))
    
    for camera_index, calibration in enumerate(calibration_parameters):
        projected_points[camera_index] = fromWorld2Img(
            points.T, **calibration).T

    return projected_points.cpu()