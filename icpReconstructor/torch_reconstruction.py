# -*- coding: utf-8 -*-
"""
This script contains a PyTorch-based implementation of the continuum robot reconstruction algorithm presented in xxx.
The package needs a PyTorch build with LAPACK/BLAS, the easiest way for getting this is by installing PyTorch using Conda/Anaconda.
"""

import torch
import torch.nn as nn
from .utils import fromWorld2Img, ball_tree_norm, PixelDataset
from abc import ABC, abstractmethod
from torchdiffeq import odeint as odeint
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
from warnings import warn
from copy import deepcopy
import numpy as np


class TorchParameterizedFunction(nn.Module, ABC):
    """
        Class for storing parameterized functions for usage with TorchMovingFrame.
        Can also be used for other models like polynomials, but it's original purpose was the
        usage as curvature polynomials.
    """

    @abstractmethod
    def get_u_fun( self, idx, s0, s1 ):
        pass


class Polynomial3Torch(TorchParameterizedFunction):
    r"""
        Parameterization of degree 3 polynomials. The parameters are the function value and the derivative at start and end.
        This, for example, allows for easy initialization given simulation results from physics-based models.

        .. note::
            The parameters for the polynomials are given as
            :math:`up(i,:) = [f(l[i-1]), f'(l[i-1]), f(l[i]), f'(l[i])]`

        Arguments
        ----------
        n : int
            Number of polynomials used to approximate the cr. Typically set to the number of tubes.

        u_p : str or n-by-4 tensor
            Optional initialization of the curvature parameters.

        continuous : boolean
            If True, the polynomials are forced to be continuous.

        random_init : boolean
            If True and u_p is None, the parameters are initialized randomly.

        end_no_curvature : boolean
            If True, the last polynomial is forced to be 0. This allows for physically more reasonable uz, so
            :math:`f_n(s) = 0`
    """

    def __init__( self, n, u_p=None, optimize=True, continuous=False, random_init=False, end_no_curvature=False ):
        super().__init__()
        self.n = n
        if u_p is not None:
            _u_p = u_p.clone().detach().reshape(-1,4)
        if continuous and optimize:
            n_f = n - end_no_curvature * 2 + 1
            n_f_dot = n - end_no_curvature
            if u_p is None:
                self.u_params = nn.Parameter(10 * (torch.rand(n_f, 1) - 0.5) if random_init else torch.zeros(n_f, 1))
                self.u_dot_params = nn.Parameter(10 * (torch.rand(n_f_dot, 2) - 0.5) if random_init else torch.zeros(n_f_dot, 2))
                u_node = torch.concatenate((self.u_params, torch.zeros(2, 1)), 0) if end_no_curvature else self.u_params
                u_dot_node = torch.concatenate((self.u_dot_params, torch.zeros(1, 2)), 0) if end_no_curvature else self.u_dot_params
            else:
                self.u_params = nn.Parameter(torch.cat((_u_p[:, 0, None], _u_p[-1, 2, None, None]), 0))
                self.u_dot_params = nn.Parameter(_u_p[:, 1::2])
                u_node = torch.concatenate((self.u_params, torch.zeros(2, 1)), 0) if end_no_curvature else self.u_params
                u_dot_node = torch.concatenate((self.u_dot_params, torch.zeros(1, 2)), 0) if end_no_curvature else self.u_dot_params
            self.u_p = torch.concatenate((u_node[:-1, :], u_dot_node[:, 0, None], u_node[1:, :], u_dot_node[:, 1, None]), 1)
        else:
            if u_p is None and optimize:
                self.u_p = nn.Parameter(10 * (torch.rand(n, 4) - 0.5) if random_init else torch.zeros(n, 4))
            elif u_p is None and not optimize:
                self.register_buffer('u_p', torch.zeros(n, 4))
            elif optimize:
                self.u_p = nn.Parameter(_u_p)
            else:
                self.register_buffer('u_p', _u_p)

    def get_u_fun( self, idx, s0, s1 ):
        r"""
            Function for generating the degree 3 polynomial from the curvature parameters.
            u_p = [f(s0), f'(s0), f(s1), f'(s1)]
        """
        u_p = self.u_p[idx, :]
        T = s1 - s0
        a = (2 * u_p[0] - 2 * u_p[2] + T * u_p[1] + T * u_p[3]) / (T ** 3)
        b = -(3 * u_p[0] - 3 * u_p[2] + 2 * T * u_p[1] + T * u_p[3]) / (T ** 2)
        c = u_p[1]
        d = u_p[0]

        def u_fun( s ):
            return torch.stack([d, c, b, a]) @ ((s - s0) ** (torch.arange(0, 4)).reshape(4, 1))

        return u_fun


class PolynomialKTorch(TorchParameterizedFunction):
    r"""
        Parameterization of degree K polynomials for the curvatures used by TorchMovingFrame.

        Arguments
        ----------
        n : int
            Number of polynomials used to approximate the cr. Typically set to the number of tubes.

        K : int
            Degree used for the polynomials.

        u_p : str or n-by-K tensor
            Optional initialization of the curvature parameters.

        optimize : boolean
            If True, the parameters are optimized during the fitting process.

        random_init : boolean
            If True and u_p is None, the parameters are initialized randomly.

        end_no_curvature : boolean
            If True, the last polynomial is forced to be 0. This allows for physically more reasonable uz, so
    """

    def __init__( self, n, K, u_p=None, optimize=True, random_init=False, end_no_curvature=False ):
        super().__init__()
        self.n = n
        self.K = K
        if u_p is not None:
            _u_p = u_p.clone().detach()
        if optimize:
            n_f = n - end_no_curvature
            if u_p is None:
                self.u_params = nn.Parameter(10 * (torch.rand(n_f, K + 1) - 0.5) if random_init else torch.zeros(n_f, K + 1))
            else:
                self.u_params = nn.Parameter(_u_p[:n_f, :])
            self.u_p = torch.concatenate((self.u_params, torch.zeros(1, K + 1)), 0) if end_no_curvature else self.u_params
        else:
            if u_p is None:
                self.register_buffer('u_p', torch.zeros(n, K + 1))
            else:
                self.register_buffer('u_p', _u_p)

    def get_u_fun( self, idx, s0, s1 ):
        '''
            Function for generating the K-th-ordner polynomial from the curvature parameters.
        '''

        def u_fun( s ):
            return self.u_p[idx, :] @ (((s - s0)/(s1 - s0)) ** (torch.arange(0, self.K + 1)).reshape(self.K + 1, 1))

        return u_fun

def __curvature_matrix_rotm( ux, uy, uz ):
    null = torch.zeros([1])
    elems = [[null, null, uy], [uz, null, null], [null, ux, null]]
    curvature_mat = torch.stack([torch.concatenate(i, 0) for i in elems], 0)
    return curvature_mat - curvature_mat.T

def __curvature_matrix_quat( ux, uy, uz ):
    null = torch.zeros([1])
    elems = [[null, null, null, null], [ux, null, uz, null], [uy, null, null, ux],[uz, uy, null, null]]
    curvature_mat = torch.stack([torch.concatenate(i, 0) for i in elems], 0)
    return curvature_mat - curvature_mat.T

CURVATURE_FCN = {'rotm':__curvature_matrix_rotm, 'quat':__curvature_matrix_quat}

class TorchPolynomialCurve(nn.Module):
    """
        NOTE: It is not recommended to use this class. It was developed for research purposes, so it is kept for reference.
        This class implements the a direct polynomial formulation for reconstructing a continuum robot's backbone.
        The parameterization of the curvatures ux, uy, and uz is achieved by providing TorchParameterizedFunction objects.

        Arguments
        ----------
        l : tensor
            List of the lengths up to which the corresponding polynomial is used. Is
            typically selected to be each NiTi tube's length outside of the actuation
            unit.

        p0 : 3-by-1 tensor
            A 3D point where the CR exits the actuation unit in world
            coordinates.

        p0_parametric : boolean
            If True, the initial position p0 is estimated as well.

        ux : n-by-4 tensor or TorchParameterizedFunction
            Initial guess for the parameters of the polynomials
            for the x-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        uy : n-by-4 tensor or TorchParameterizedFunction
            Initial guess for the parameters of the polynomials
            for the y-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        uz : n-by-4 tensor or TorchParameterizedFunction
            Initial guess for the parameters of the polynomials
            for the z-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        optimize : list of 3 booleans
            If True, the corresponding curvature [ ux, uy, uz ] is optimized during the fitting process. Only used if the 
            corresponding curvature is not given as a TorchParameterizedFunction object.

        continuous_u : list of 3 booleans
            If true, the corresponding curvature [ ux, uy, uz ] is enforced to be continuous. Only used if the 
            corresponding curvature is not given as a TorchParameterizedFunction object.

        random_init : list of 3 booleans
            If true, the corresponding curvature [ ux, uy, uz ] is randomly initialized. Only used if the 
            corresponding curvature is not given as a TorchParameterizedFunction object.

    """
    
    def __init__(
            self, l, p0=None, p0_parametric=False, 
            ux=None, uy=None, uz=None, optimize=[True, True, True], continuous_u=[True, True, True], 
            random_init=[False] * 3 
            ):
        super().__init__()
        
        self.ux_funs, self.uy_funs, self.uz_funs = None, None, None

        self.l = l
        self.n = len(l)
        
        if p0 is None:
            p0 = torch.zeros(3, 1)

        if p0_parametric:
            self.p0 = nn.Parameter(p0)
        else:
            self.p0 = p0
            
        if issubclass(type(ux), TorchParameterizedFunction):  # Check if ux is already a TorchParameterizedFunction.
            self.ux = ux
        else:  # If not, initialize it as a Polynomial3Torch.
            self.ux = Polynomial3Torch(self.n, ux, optimize=optimize[0], continuous=continuous_u[0], random_init=random_init[0])

        if issubclass(type(uy), TorchParameterizedFunction):  # Check if uy is already a TorchParameterizedFunction.
            self.uy = uy
        else:  # If not, initialize it as a Polynomial3Torch.
            self.uy = Polynomial3Torch(self.n, uy, optimize=optimize[1], continuous=continuous_u[1], random_init=random_init[1])

        if issubclass(type(uz), TorchParameterizedFunction):  # Check if uz is already a TorchParameterizedFunction.
            self.uz = uz
        else:  # If not, initialize it as a Polynomial3Torch.
            self.uz = Polynomial3Torch(self.n, uz, optimize=optimize[2], continuous=continuous_u[2], random_init=random_init[2])

        self.set_funs()
        
    def set_funs( self ):
        """
            Function that fills the curvature function handles. Has to be called after each optimizer step.
        """
        self.ux_funs = [self.ux.get_u_fun(0, 0, self.l[0])]
        self.uy_funs = [self.uy.get_u_fun(0, 0, self.l[0])]
        self.uz_funs = [self.uz.get_u_fun(0, 0, self.l[0])]

        for i in range(1, self.n):
            self.ux_funs.append(self.ux.get_u_fun(i, self.l[i - 1], self.l[i]))
            self.uy_funs.append(self.uy.get_u_fun(i, self.l[i - 1], self.l[i]))
            self.uz_funs.append(self.uz.get_u_fun(i, self.l[i - 1], self.l[i]))
            
    def forward( self, s_val ):
        """
        TODO: Something about this is off, but I cannot figure out what it is.
        """
        for i in range(len(self.l)):
            s = s_val[(s_val <= self.l[i]) & ((self.l[i-1] if i >= 1 else 0) <= s_val)]
            if i == 0:
                p = torch.stack([self.ux_funs[i](s), self.uy_funs[i](s), self.uz_funs[i](s)], 0)
            else:
                p = torch.stack([self.ux_funs[i](s), self.uy_funs[i](s), self.uz_funs[i](s)], 0)
        
        return p.T


class TorchMovingFrame(nn.Module):
    """
        This class implements the moving frame formulation for reconstructing a continuum robot's backbone.
        The parameterization of the curvatures ux, uy, and uz is achieved by providing TorchParameterizedFunction objects.

        Arguments
        ----------
        l : tensor
            List of the lengths up to which the corresponding polynomial is used. Is
            typically selected to be each NiTi tube's length outside of the actuation
            unit.

        p0 : 3-by-1 tensor
            A 3D point where the CR exits the actuation unit in world
            coordinates.

        p0_parametric : boolean
            If True, the initial position p0 is estimated as well.

        R0 : 3-by-3 or 4-by-1 tensor
            Either a 3-by-3 matrix from SO(3) that describes the orientation of
            the exiting CR (ref. [1]) or, if R0_parametric is True, tensor of
            shape (4,). The matrix is then parametrized using quaternions.

        R0_parametric : boolean
            If True, the initial orientation R0 is estimated as well. To ensure
            that R0 remains inside SO(3), it is parameterized using quaternions.

        ux : n-by-4 tensor or TorchParameterizedFunction
            Initial guess for the parameters of the polynomials
            for the x-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        uy : n-by-4 tensor or TorchParameterizedFunction
            Initial guess for the parameters of the polynomials
            for the y-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        uz : n-by-4 tensor or TorchParameterizedFunction
            Initial guess for the parameters of the polynomials
            for the z-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        optimize : list of 3 booleans
            If True, the corresponding curvature [ ux, uy, uz ] is optimized during the fitting process. Only used if the 
            corresponding curvature is not given as a TorchParameterizedFunction object.

        integrator : str
            Integration method from the following list:

                - "midpoint"
                - "rk4"
                - "dopri5",
                - "bost3"
                - "fehlberg2"
                - "adaptive_heun"
                - "euler"
                - "dopri8"
                - "explicit_adams"
                - "implicit_adams"

            The midpoint rule typically gives a good trade-off between accuracy and speed.

        continuous_u : list of 3 booleans
            If true, the corresponding curvature [ ux, uy, uz ] is enforced to be continuous. Only used if the 
            corresponding curvature is not given as a TorchParameterizedFunction object.

        random_init : list of 3 booleans
            If true, the corresponding curvature [ ux, uy, uz ] is randomly initialized. Only used if the 
            corresponding curvature is not given as a TorchParameterizedFunction object.

        Sources
        ----------
        [1] 10.1109/TRO.2010.2062570, Rucker, D. Caleb, Bryan A. Jones, and Robert J. Webster III. "A geometrically exact model for externally loaded concentric-tube continuum robots." IEEE transactions on robotics 26.5 (2010): 769-780.
    """

    def __init__(
            self, l, p0=None, p0_parametric=False, R0=None, R0_parametric=False, rotation_method="rotm", 
            ux=None, uy=None, uz=None, optimize=[True, True, False], integrator='midpoint', continuous_u=[False, False, True], 
            random_init=[False] * 3 
            ):
        super().__init__()

        self.__integrator = 'midpoint'
        self.__previous_integrate_parameters = None
        self.ux_funs, self.uy_funs, self.uz_funs = None, None, None
        self.rotation_method = rotation_method

        self.l = l
        self.n = len(l)

        if p0 is None:
            p0 = torch.zeros(3, 1)

        if p0_parametric:
            self.p0 = nn.Parameter(p0)
        else:
            self.p0 = p0
        
        if R0 is None:
            if rotation_method == "quat" or R0_parametric:
                R0 = torch.tensor([1, 0, 0, 0])
            else:
                R0 = torch.eye(3)
        
        if R0_parametric:
            # If R0 is parametric, it is assumed to be a quaternion. The quaternion is then normalized and converted to a rotation matrix.
            if R0.shape != torch.Size([4, ]):
                raise Exception("If R0_parametric is set to true, the input R0 is considered to be a quaternion, so its shape is supposed to be (4,).")
            _R0 = R0.float()
            self.q = nn.Parameter(_R0 / torch.linalg.vector_norm(_R0))
            q = self.q / torch.linalg.vector_norm(self.q)  # Setup the quaternion.
            
            if rotation_method == "quat":
                self.R0 = q
            else:
                self.R0 = torch.zeros((3, 3))
                # Convert the quaternion to a rotation matrix R0.
                self.R0[0, 0] = 1 - 2 * (q[2] ** 2 + q[3] ** 2)
                self.R0[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
                self.R0[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])
                self.R0[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
                self.R0[1, 1] = 1 - 2 * (q[1] ** 2 + q[3] ** 2)
                self.R0[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])
                self.R0[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
                self.R0[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
                self.R0[2, 2] = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        else:
            self.R0 = R0
        
        self.x0 = torch.cat((self.p0.flatten(), self.R0.flatten()), 0)
        self.integrator = integrator
        self.ode_solution = None
        self.ode = self.__quat_ode if rotation_method == "quat" else self.__rotm_ode
        self.curvature_mat_fcn = CURVATURE_FCN[rotation_method]

        if issubclass(type(ux), TorchParameterizedFunction):  # Check if ux is already a TorchParameterizedFunction.
            self.ux = ux
        else:  # If not, initialize it as a Polynomial3Torch.
            self.ux = Polynomial3Torch(self.n, ux, optimize=optimize[0], continuous=continuous_u[0], random_init=random_init[0])

        if issubclass(type(uy), TorchParameterizedFunction):  # Check if uy is already a TorchParameterizedFunction.
            self.uy = uy
        else:  # If not, initialize it as a Polynomial3Torch.
            self.uy = Polynomial3Torch(self.n, uy, optimize=optimize[1], continuous=continuous_u[1], random_init=random_init[1])

        if issubclass(type(uz), TorchParameterizedFunction):  # Check if uz is already a TorchParameterizedFunction.
            self.uz = uz
        else:  # If not, initialize it as a Polynomial3Torch.
            self.uz = Polynomial3Torch(self.n, uz, optimize=optimize[2], continuous=continuous_u[2], random_init=random_init[2])

        self.set_funs()  # Fill the curvature function handles.

    def set_funs( self ):
        """
            Function that fills the curvature function handles. Has to be called after each optimizer step.
        """
        self.ux_funs = [self.ux.get_u_fun(0, 0, self.l[0])]
        self.uy_funs = [self.uy.get_u_fun(0, 0, self.l[0])]
        self.uz_funs = [self.uz.get_u_fun(0, 0, self.l[0])]

        for i in range(1, self.n):
            self.ux_funs.append(self.ux.get_u_fun(i, self.l[i - 1], self.l[i]))
            self.uy_funs.append(self.uy.get_u_fun(i, self.l[i - 1], self.l[i]))
            self.uz_funs.append(self.uz.get_u_fun(i, self.l[i - 1], self.l[i]))

    @property
    def integrator( self ):
        return self.__integrator

    @integrator.setter
    def integrator( self, value ):
        if value not in ["midpoint", "rk4", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "dopri8", "explicit_adams", "implicit_adams"]:
            raise Exception(
                "The given integrator is not valid or available. Please use one of the solvers given by the torchdiffeq package: https://github.com/rtqichen/torchdiffeq")
        if value != self.__integrator:
            """
                Ensure that the integration is run again after the integrator was changed.
            """
            self.__previous_integrate_parameters = None
        self.__integrator = value

    def u_hat( self, s ):
        """
            Function for transforming u_p to the necessary skew-symmstric matrix. idx_fun is the index of the curvature function that is active at the current
            arc-length.
        """
        if s <= self.l[-1]:
            active_tube_idx = torch.where(self.l >= s)[0]
            idx_fun = torch.min(active_tube_idx)
        else:
            idx_fun = self.n - 1

        ux, uy, uz = self.ux_funs[idx_fun](s), self.uy_funs[idx_fun](s), self.uz_funs[idx_fun](s)
        curvature_mat = self.curvature_mat_fcn( ux, uy, uz )
        return curvature_mat
    
    def forward( self, s_val ):
        return self.integrate(s_val)[:,:3]
    
    def integrate( self, s_val ):
        """
            Get the solution of the integration. Integration is only performed if the previous (curvature) parameters did change or if
            the integration scheme changed.

            Arguments:
            ----------
            s_val : tensor, optional
                If values for the arc-length are given, the integration scheme is evaluated at these points and not the
                ones used in the integration. This is useful for plotting the solution. If None, the arc-lengths used in the integration.
        """
        if self.__previous_integrate_parameters is None or (list(self.__previous_integrate_parameters()) == list(self.parameters())):            
            self.ode_solution = odeint(self.ode, self.x0, s_val, method=self.integrator)
        return self.ode_solution
       
    def __quat_ode( self, s, x ):
        """
            Right-hand side of the quaternion moving frame ODE.

            Arguments
            ----------
            s : float
                The current value of the arc-length.

            x : 7-by-1 float tensor
                The current state x = [p, q]. p is the position of the CR in world coordinates and q is the quaternion describing the orientation of the
                CR at the current arc-length.
        """
        q = x[3:]
        dp1 = (q[1]*q[3]+q[0]*q[2])
        dp2 = (q[2]*q[3]-q[0]*q[1])
        dp3 = (-q[1]**2-q[2]**2)
        dp = torch.tensor([0,0,1])+2*torch.stack([dp1, dp2, dp3])/(torch.sum(q**2))
        dq = self.u_hat(s)@q/2
        return torch.concat((dp, dq), 0)
    
    def __rotm_ode( self, s, x ):
        """
            Right-hand side of the moving frame ODE.

            Arguments
            ----------
            s : float
                The current value of the arc-length.

            x : 12-by-1 float tensor
                The current state x = [p, R]. p is the position of the CR in world coordinates and R is the rotation matrix describing the orientation of the
                CR at the current arc-length.
        """
        R = x[3:].reshape((3, 3))

        dp = R[:, 2].flatten()
        dR = (R @ self.u_hat(s)).flatten()
        return torch.concat((dp, dR), 0)
    

class TorchCurveEstimator():
    r"""
        This class implements the PyTorch-based curve estimator (TCE) for continuum robots.
        The TCE takes a parameterized model for the continuum robot - this can be a rotation
        frame, circular arcs, differentiable rendering model, ... - and optimizes for the
        torch.parameters using an Iterative Closest Point algorithm.

        Arguments
        ----------
        curve_model : nn.Module
            PyTorch model that parameterizes the reconstruction of thecontinuum robot. The model has to be
            callable with the arc-length as the only argument.
        
        l : positive scalar
            The length of the continuum robot from the starting position to the tip of the robot.

        camera_calibration_parameters : list of dictionaries
            A list of dictionaries containing the keys
            "A", "dist", "P", "R", and "T". The function "camera_folder_to_params" can
            automatically generate this list for you.

        n_steps : int
            Number of intermediate steps to evaluate the reconstructed cr at.
            If low, the computation is fast, but for very low values the convergence might
            be bad. If high, a better solution might be found, at cost of higher computation
            times.

        w : double
            Weighting of the two cost functions. 0 (only track distance of pixels to
            reconstruction) is typically a good value, but if parts of the cr are occluded,
            increasing w can help.

        dist_norm : int
            Order of the metric used to penalize distances. Higher values give more cost to big errors.
    """
    
    def __init__(self, curve_model, camera_calibration_parameters, l, w=0., n_steps=50, dist_norm=2, post_step_cb=[], post_epoch_cb=[] ):
        super().__init__()
        assert isinstance(curve_model, nn.Module), "The curve_model needs to be an instance of nn.Module" 
        self.camera_calibration_parameters = camera_calibration_parameters
        self.curve_model = curve_model
        self.l = l
        
        self.w = w
        self.n_steps = n_steps
        self.dist_norm = dist_norm
        self.pixel_diff_state = None
        self.backbone_diff_state = None
        self._bb_pixel_coordinates = None
        if len(camera_calibration_parameters) < 2:
            warn("Two or more images are necessary for a successful reconstruction.")
        self.camera_calibration_parameters = camera_calibration_parameters
        self.s_val = torch.linspace(0, self.l, self.n_steps)
        self.reset_diff_states()
        self.post_step_cb = post_step_cb
        self.post_epoch_cb = post_epoch_cb

    def get_img_coordinates( self ):
        """
            Project the computed backbone points to all the images and return the image coordinates.
        """
        if self._bb_pixel_coordinates is None:
            backbone_pts = self.curve_model(self.s_val).T
            self._bb_pixel_coordinates = []
            for i in range(len(self.camera_calibration_parameters)):
                self._bb_pixel_coordinates.append(fromWorld2Img(backbone_pts, **self.camera_calibration_parameters[i]).T)
        return self._bb_pixel_coordinates

    def reset_diff_states( self ):
        """
            Reset the diff states, so that in the next call off get_*_diff_idx, the correspondances are computated agian.
        """
        self.pixel_diff_state = None
        self.backbone_diff_state = None

    def get_pixel_diff_idx( self, cr_img_coordinates, img_idx_data ):
        """
            Computes for each CR pixel the index of the closest reconstruction pixel.

            Arguments
            ----------
            cr_img_coordinates : m-by-2 int
                Coordinates of CR pixels in image coordinates.

            img_idx_data : m-by-1 int
                Vector containing the index from which image the corresponding img_coordinates entry was taken.
        """
        backbone_img_coordinates = self.get_img_coordinates()
        if self.pixel_diff_state is None:
            idx_min_dist = []
            cr_img_coordinates_sorted = []
            for i in torch.unique(img_idx_data):
                cr_img_coordinates_i = cr_img_coordinates[img_idx_data == i, :]
                # _, idc_min = brute_force_distance_norm(cr_img_coordinates_i.float(), backbone_img_coordinates[i], p=self.dist_norm)
                idc_min = ball_tree_norm(cr_img_coordinates_i.float(), backbone_img_coordinates[i], p=self.dist_norm)
                idx_min_dist.append(idc_min)
                cr_img_coordinates_sorted.append(cr_img_coordinates_i)
            self.pixel_diff_state = dict(idx_min_dist=idx_min_dist, cr_img_coordinates_sorted=cr_img_coordinates_sorted)
        
        return self.pixel_diff_state["idx_min_dist"], self.pixel_diff_state["cr_img_coordinates_sorted"]

    def pixel_diff( self, cr_img_coordinates, img_idx_data ):
        """
            Compute the distance of each CR pixel to the closest reconstruction point in image coordinates.

            Arguments
            ----------
            cr_img_coordinates : m-by-2 tensor
                Tensor containing the image coordinates of CR pixels.

            img_idx_data :m-by-1 tensor
                Tensor containing the index of the image a point was taken from.
       """
        curve_pixel_coordinates = self.get_img_coordinates()
        diffs = None
        idx_min_dist, cr_img_coordinates_sorted = self.get_pixel_diff_idx(cr_img_coordinates, img_idx_data)
        for i in torch.unique(img_idx_data):
            diffs = curve_pixel_coordinates[i][idx_min_dist[i], :] - cr_img_coordinates_sorted[i] if diffs is None else torch.concatenate(
                (diffs, curve_pixel_coordinates[i][idx_min_dist[i], :] - cr_img_coordinates_sorted[i]), 0)
        return diffs

    def pixel_loss( self, cr_img_coordinates, img_idx_data ):
        """
            Computes the average distance of each CR pixel to the corresponding backbone point.
        """
        # Penalizes the minimum distance of each measurement point to the reconstructed curve.
        return torch.linalg.vector_norm(self.pixel_diff(cr_img_coordinates, img_idx_data), self.dist_norm, 1).mean()  # **self.dist_norm

    def get_backbone_diff_idx( self, cr_img_coordinates, img_idx_data ):
        """
            Computes for each backbone point in image coordinates the index of the closest CR pixel .

            Arguments
            ----------
            cr_img_coordinates : m-by-2 int
                Coordinates of CR pixels in image coordinates.

            img_idx_data : m-by-1 int
                Vector containing the index from which image the corresponding img_coordinates entry was taken.

        """
        backbone_img_coordinates = self.get_img_coordinates()
        if self.backbone_diff_state is None:
            idx_min_dist = []
            cr_img_coordinates_sorted = []
            for i in torch.unique(img_idx_data):
                cr_img_coordinates_i = cr_img_coordinates[img_idx_data == i, :]
                # _, idc_min = brute_force_distance_norm(backbone_img_coordinates[i], cr_img_coordinates_i.float(), p=self.dist_norm)
                idc_min = ball_tree_norm(backbone_img_coordinates[i], cr_img_coordinates_i.float(), p=self.dist_norm)
                idx_min_dist.append(idc_min)
                cr_img_coordinates_sorted.append(cr_img_coordinates_i)
            self.backbone_diff_state = dict(idx_min_dist=idx_min_dist, cr_img_coordinates_sorted=cr_img_coordinates_sorted)
        return self.backbone_diff_state["idx_min_dist"], self.backbone_diff_state["cr_img_coordinates_sorted"]

    def backbone_diff( self, cr_img_coordinates, img_idx_data ):
        """
            Compute the distance of each CR pixel to the closest reconstruction point in image coordinates.

            Arguments
            ----------
            cr_img_coordinates : m-by-2 tensor
                Tensor containing the image coordinates of CR pixels.

            img_idx_data :m-by-1 tensor
                Tensor containing the index of the image a point was taken from.
       """
        curve_pixel_coordinates = self.get_img_coordinates()
        diffs = None
        idx_min_dist, cr_img_coordinates_sorted = self.get_backbone_diff_idx(cr_img_coordinates, img_idx_data)
        for i in torch.unique(img_idx_data):
            diffs = curve_pixel_coordinates[i] - cr_img_coordinates_sorted[i][idx_min_dist[i], :] if diffs is None else torch.concatenate(
                (diffs, curve_pixel_coordinates[i] - cr_img_coordinates_sorted[i][idx_min_dist[i], :]), 0)
        return diffs

    def backbone_loss( self, img_coordinates, img_idx_data ):
        """
            Computes the average distance of each backbone point to the corresponding CR pixel.
        """
        # Penalizes the minimum distance of each measurement point to the reconstructed curve.
        return torch.linalg.vector_norm(self.backbone_diff(img_coordinates, img_idx_data), self.dist_norm, 1).mean()  # **self.dist_norm

    def loss( self, img_coordinates, img_idx_data ):
        """
            Combines the two cost as a weighted sum using w as the weight.

            Arguments
            ----------
            img_coordinates : m-by-2 tensor
                Tensor containing the image coordinates of CR pixels.

            img_idx_data :m-by-1 tensor
                Tensor containing the index of the image a point was taken from.
       """
        self._bb_pixel_coordinates = None

        loss = 0
        if self.w < 1.:
            loss += self.pixel_loss(img_coordinates, img_idx_data) * (1 - self.w)
        if self.w > 0.:
            loss += self.backbone_loss(img_coordinates, img_idx_data) * self.w
        return loss

    def loss_3d_dist( self, s, pts ):
        """
            Cost function for a labeled set of 3D points with their corresponding arc-length. The weight w is used to
            weight the two cost functions, average distance of the backbone points to the 3D points and distance of the
            tip of the cr to the last 3D point.

            Arguments
            ----------
            s : m-by-1 tensor
                Tensor of (approximated) arc-lengths.

            pts : m-by-3 tensor
                3D points of the cr that are known.
        """
        pt_idc_0 = torch.argmin(torch.abs(self.s_val.reshape(-1, 1) - s.reshape(1, -1)), 0)
        backbone_pts = self.curve_model(self.s_val)
        return ((1 - self.w) * torch.linalg.vector_norm(backbone_pts[pt_idc_0, :] - pts, self.dist_norm, 1).mean() + self.w * torch.linalg.vector_norm(
            pts[-1, :] - pts[-1, :], self.dist_norm, 0))

    def fit( self, pixel_list, optimizer=None, n_iter=5, batch_size=None, lr=0.2, repetitions=1, grad_tol=1e-4, scheduler=None, verbose=False, patience=5, tol=-1e-4 ):
        """
            Run the optimization given the the image coordinates of the CR. Per default, an Adam optimizer is used, the full dataset is used and the point correspondances
            are reset after each gradient descent step.

            Arguments
            ----------
            optimizer : PyTorch optimizer
                Adam with a small weight decay value (1e-6) is recommended.

            pixel_list : list
                List of pixels corresponding to the CR in each image. The length has to match
                the number of camera calibration parameters.

            n_iter : int
                Number of epochs, so how often the data set is used for optimization.

            lr : double
                Learning rate of the default optimizer.

            batch_size : int
                Number of CR pixels used per Stochastic Gradient Descent step.

            repetitions : int
                Number how often the same batch is reused, reducing the need for recalculating the point correspondances.

            grad_tol : double
                Tolerance on the necessary condition of optimality. If the abolsute value of all partial derivatives is below grad_tol, the algorithm terminates
                early.
        """
        if len(pixel_list) != len(self.camera_calibration_parameters):
            raise Exception("The list containing the pixel positions of the CR has to match in length the number of given camera calibration parameters.")

        if optimizer is None:
            optimizer = torch.optim.Adam(self.curve_model.parameters(), lr, weight_decay=1e-6)
            
        use_scheduler = scheduler is not None

        dataset = PixelDataset(pixel_list)
        
        loss_hist = []

        if batch_size is None:
            batch_size = int(len(dataset))
        
        rndsampler = RandomSampler(dataset)
        batchsampler = BatchSampler(rndsampler, batch_size=batch_size, drop_last=True)
        dataloader = DataLoader(dataset, sampler=batchsampler)

        with torch.no_grad():
            loss_hist.append(self.loss(dataset.p, dataset.img_idx_data).detach().cpu().numpy())
            lowest_loss = loss_hist[-1]
            best_model = deepcopy(self.curve_model.state_dict())
        
        for epoch in range(1, n_iter + 1):
            with tqdm(enumerate(dataloader), disable=not verbose) as pbar:
                for iter, (smpl, idc) in pbar:
                    smpl = smpl.squeeze()
                    idc = idc.squeeze()
                    if smpl.shape[0] < batch_size:
                        continue
                    for repetition in range(repetitions):
                        optimizer.zero_grad()
                        self._bb_pixel_coordinates = None
                        loss = self.loss(smpl, idc)
                        loss.backward()
                        if verbose:
                            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                            pbar.set_description(f"Epoch {epoch}/{n_iter}, Iteration {iter + 1}, Repetition {repetition + 1}/{repetitions}")
                        optimizer.step()
                        
                        self.curve_model.set_funs()
                        for f in self.post_step_cb:
                            f(self)
                        # if ~torch.any(torch.tensor([torch.any(torch.abs(i.grad) >= grad_tol) for i in self.curve_model.parameters()]).bool()):
                        #     break
                    self.reset_diff_states()
            for f in self.post_epoch_cb:
                f(self)
            if use_scheduler:
                scheduler.step()
            with torch.no_grad():
                loss_hist.append(self.loss(dataset.p, dataset.img_idx_data).detach().cpu().numpy())
            if loss_hist[-1] < lowest_loss:
                best_model = deepcopy(self.curve_model.state_dict())
                lowest_loss = loss_hist[-1]
            # Early stopping if the loss did not decrease in the last patience epochs.
            if epoch > patience and (lowest_loss - loss_hist[-patience]) / abs(lowest_loss) > tol:
                break
        self.curve_model.load_state_dict(best_model)
        return np.stack(loss_hist)
