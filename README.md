# Iterative Closest Point Reconstructor for Continuum Robots

This repository provides the implementations and examples used in our publication "An iterative closest point algorithm for marker-free 3D shape registration of continuum robots". All of the implementations were done in Python.

## Overview

This project arose from our work "An iterative closest point algorithm for marker-free 3D shape registration of continuum robots", available on [arXiv](https://arxiv.org/abs/2405.15336), on the shape estimation of continuum robots (CR), more precisely concentric tube continuum robots we plan on using for minimally invasive neurosurgery procedures. The reconstruction of CRs involves finding the backbone -- the central line -- of the robot.

Here, we aim to provide tools and methods for reproducing the results from our work and to enable other's to find their robot's shape. This package is built up in an object-oriented manner, so that user's can easily implement compatible sub-modules, backbone models and algorithms

## Installation
This package is available via PyPI:
```
pip install icpReconstructor
```
It requires the following packages:
- PyTorch
- CasADi
- NumPy
- torchdiffeq     
- scikit-learn
- scikit-image
- tqdm
- NetworkX
- SciPy


## Prerequisites

This package requires you to provide the parameters of a camera-calibration, in particular the following:
- A: The cameras' intrinsic matrices (3x3) including focal lengths and principal point.
- dist: The distortion coefficients (k1, k2, p1, p2, k3) for radial and tangential distortion.
- P: The projection matrix (3x4) used to project 3D camera coordinates onto the image plane.
- R: The rotation matrix (3x3) describing the orientation of the first camera in world coordinates.
- T: The translation vector (3x1) describing the position of the first camera in world coordinates.

We chose these formulations to be in line with the camera calibration implementation of [OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) for stereo calibration. A tutorial can also the found [there](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

## Documentation

For in-depth information about the algorithms and APIs used in CTCR reconstruction, refer to our [Github Wiki](https://github.com/MKHoffmann/icpReconstructor/wiki) and our work.

## Examples

Within this repository, we provide a set of examples on one set of binary images of a simulated concentric tube continuum robot. These include the One-Step and Multi-Step algorithms presented in our work, but also different ways of warmstarting using image-processing algorithms and space-carving.

## How to Cite

If you find our project valuable for your research or work, please consider citing it using the following format:
Hoffmann, M., Mühlenhoff, J., Ding, Z., Sattel T. , Flaßkamp, K., "An iterative closest point algorithm for marker-free 3D shape registration of continuum robots" arXiv preprint arXiv:2405.15336 (2024).