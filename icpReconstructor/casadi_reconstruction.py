import casadi
import numpy as np
from abc import ABC, abstractmethod
from .utils import discretizeODE, fromWorld2ImgCasadi


class ParameterFunctionCasadi(ABC):
    """
        Class for storing parameterized functions for usage with Casadi. Allows for fast initialization
        of corresponding PyTorch modules.
    """

    @abstractmethod
    def get_u_fun( self, idx, s0, s1 ):
        pass
    
    @abstractmethod
    def get_parameters( self ):
        # Method that outputs the symbolic parameters (decision variables) of the parameterized function.
        pass


class Polynomial3Casadi(ParameterFunctionCasadi):
    r"""
       Parameterization of third-order polynomials. The parameters are the function value and the derivative at start and end.
       This, for example, allows for easy initialization given simulation results from physics-based models.

       .. note::
           The parameters for the polynomials are given as
           :math:`up(i,:) = [f(l[i-1]), f'(l[i-1]), f(l[i]), f'(l[i])]`

       Arguments
       ----------
       n : int
           Number of polynomials used to approximate the ctcr. Typically set to the number of tubes.

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

    def __init__( self, opt, n, u_p=None, optimize=True, continuous=False, random_init=False, end_no_curvature=False ):
        super().__init__()
        self.n = n
        self.optimize = optimize
        if not (u_p is None) and type(u_p) not in [casadi.MX, np.array, np.ndarray]:
            raise Exception("The provided curvature parameters should either be casadi.MX objects or numpy arrays.")
        if continuous and optimize:
            n_f = n - end_no_curvature * 2 + 1
            n_f_dot = n - end_no_curvature
            if u_p is None:
                self.u_params = opt.variable(n_f, 1)
                self.u_dot_params = opt.variable(n_f_dot, 2)
                if random_init:
                    opt.set_initial(self.u_params, 10 * (np.random.randn(n_f, 1) - 0.5))
                    opt.set_initial(self.u_dot_params, 10 * (np.random.randn(n_f_dot, 2) - 0.5))
            else:
                self.u_params = opt.variable(self.n - 1, 1)
                self.u_dot_params = opt.variable(self.n - 1, 2)
                opt.set_initial(self.u_params, u_p[:-1, 0, None])
                opt.set_initial(self.u_dot_params, u_p[:-1, 2:])
            u_node = casadi.vertcat(self.u_params, casadi.MX.zeros((2, 1))) if end_no_curvature else self.u_params
            u_dot_node = casadi.vertcat(
                self.u_dot_params, casadi.MX.zeros((1, 2))
                ) if end_no_curvature else self.u_dot_params
            self.u_p = casadi.cse(casadi.horzcat(u_node[:-1, :], u_dot_node[:, 0], u_node[1:, :], u_dot_node[:, 1]))
        else:
            if u_p is None and optimize:
                self.u_p = opt.variable(n, 4)
                opt.set_initial(self.u_p, 10 * (np.random.rand(n, 4) - 0.5) if random_init else np.zeros((n, 4)))
            elif u_p is None and not optimize:
                self.u_p = casadi.MX.zeros((n, 4))
            elif optimize:
                self.u_p = opt.variable(n, 4)
                opt.set_initial(self.u_p, u_p)
            else:
                self.u_p = casadi.MX(u_p)

    def get_u_fun( self, idx, s0, s1 ):
        '''
            Function for generating the polynomial of degree 3 from the curvature parameters.
            u = [f(s0), f'(s0), f(s1), f'(s1)]
        '''
        u = self.u_p[idx, :]
        T = s1 - s0
        a = (2 * u[0] - 2 * u[2] + T * u[1] + T * u[3]) / (T ** 3)
        b = -(3 * u[0] - 3 * u[2] + 2 * T * u[1] + T * u[3]) / (T ** 2)
        c = u[1]
        d = u[0]

        def u_fun( s ):
            return casadi.horzcat(d, c, b, a) @ ((s - s0) ** (np.arange(0, 4)).reshape(4, 1))

        return u_fun

    def get_value( self, sol ):
        """
            Returns the value of the curvature parameters as a PyTorch tensor.
        """
        return sol.value(self.u_p)
    
    def get_parameters(self):
        return self.u_p


class PolynomialKCasadi(ParameterFunctionCasadi):
    r"""
            Parameterization of degree K polynomials. This implementation uses Casadi for the optimization.

            Arguments
            ----------
            n : int
                Number of polynomials used to approximate the ctcr. Typically set to the number of tubes.

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

    def __init__( self, opt, n, K, u_p=None, optimize=True, random_init=False, end_no_curvature=False ):
        super().__init__()
        self.n = n
        self.K = K
        self.optimize = optimize
        if optimize:
            n_f = n - end_no_curvature * 2 + 1
            if u_p is None:
                self.u_params = opt.variable(n_f, K + 1)
                opt.set_initial(self.u_params, 10 * (np.random.rand(n_f, K + 1) - 0.5) if random_init else np.zeros((n_f, K + 1)))
            else:
                self.u_params = opt.variable(n_f, K + 1)
                opt.set_initial(self.u_params, u_p[:-1, :])
            u_node = casadi.vertcat(self.u_params, casadi.MX.zeros((1, K + 1))) if end_no_curvature else self.u_params
            self.u_p = casadi.horzcat(u_node[:-1, :], u_node[1:, :])
        else:
            if u_p is None:
                self.u_p = casadi.MX.zeros((n, K + 1))
            else:
                self.u_p = casadi.MX(u_p)

    def get_u_fun( self, idx, s0, s1 ):
        '''
            Function for generating the K-th-ordner polynomial from the curvature parameters.
        '''

        def u_fun( s ):
            return self.u_p[idx, :] @ (((s - s0) / (s1 - s0)) ** (np.arange(0, self.K + 1)).reshape(self.K + 1, 1))

        return u_fun
    
    def get_parameters(self):
        return self.u_p


class CasadiCurveModel(ABC):
    r"""
        Abstract base class for the curve model. This class implements the basic functionality that is needed for all curve models. The curve models
        themselves are implemented in the subclasses of this class.
    """
    
    def __init__( self ):
        self.opt = casadi.Opti()
    
    @abstractmethod
    def get_p( self ):
        # return the symbolic values for the positions along the backbone p
        pass
    
    @abstractmethod
    def setup( self ):
        # The setup function is called once at the beginning of the ICP algorithm. It is used for example for setting up the constraints of a problem.
        pass
    
    @abstractmethod
    def collect_parameters( self ):
        # Method for requesting from all parameterized functions their corresponding parameters
        pass

class CasadiRotationFrame(CasadiCurveModel):
    r"""
        This class implements the rotation frame formulation for reconstructing a continuum robot's backbone.
        The parameterization of the curvatures ux, uy, and uz is achieved by providing TorchCurvature objects.

        Arguments
        ----------
        l : tensor
            List of the lengths up to which the corresponding polynomial is used. Is
            typically selected to be each NiTi tube's length outside of the actuation
            unit.

        p0 : 3-by-1 tensor
            A 3D point where the CTCR exits the actuation unit in world
            coordinates.

        p0_parametric : boolean
            If True, the initial position p0 is estimated as well.

        R0 : 3-by-3 or 4-by-1 tensor
            Either a 3-by-3 matrix from SO(3) that describes the orientation of
            the exiting CTCR (ref. [1]) or, if R0_parametric is True, tensor of
            shape (4,). The matrix is then parametrized using quaternions.

        R0_parametric : boolean
            If True, the initial orientation R0 is estimated as well. To ensure
            that R0 remains inside SO(3), it is parameterized using quaternions.

        ux : n-by-4 tensor or TorchCurvature
            Initial guess for the parameters of the polynomials
            for the x-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        uy : n-by-4 tensor or TorchCurvature
            Initial guess for the parameters of the polynomials
            for the y-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        uz : n-by-4 tensor or TorchCurvature
            Initial guess for the parameters of the polynomials
            for the z-curvatures of the n polynomials. If set to None it is either initialized
            as 0s or uniformly distributed between -5 and 5 if random_init is True.

        integrator : str
            Integration method from the following list:

                - "rk1" / "euler"
                - "rk2" / "midpoint"
                - "heun"
                - "rk3" / "simpson"
                - "rk4"
                - "3/8"

            The rk4 typically gives a good trade-off between accuracy and speed.

        continuous_u : list of 3 booleans
            If true, the corresponding curvature [ ux, uy, uz ] is enforced to be continuous.

        random_init : list of 3 booleans
            If true, the corresponding curvature [ ux, uy, uz ] is randomly initialized.

        Sources
        ----------
        [1] 10.1109/TRO.2010.2062570, Rucker, D. Caleb, Bryan A. Jones, and Robert J. Webster III. "A geometrically exact model for externally loaded concentric-tube continuum robots." IEEE transactions on robotics 26.5 (2010): 769-780.
    """

    
    def __init__(
            self, l, p0=np.zeros((3, 1)), p0_parametric=False, R0=np.eye(3, 3), R0_parametric=False, ux=None, uy=None, uz=None,
            optimize=[True, True, False], n_steps=50, w=0.0, method='rk4', dist_norm=2, continuous_u=[False, False, True], random_init=[False] * 3
            ):
        super().__init__()
        self.opt.solver('ipopt')
        self.sol = None
        self.__method = method
        self.l = l
        self.n = len(l)
        self.n_steps = n_steps
        self.p0 = p0
        self.continuous_u = continuous_u
        self.random_init = random_init
        self.optimize = optimize
        self.function_parameters = None
        
        self.ux_funs = []
        self.uy_funs = []
        self.uz_funs = []

        if p0_parametric:
            self.p0 = self.opt.variable(3, 1)
        else:
            self.p0 = p0

        if R0_parametric:
            # If R0 is parametric, it is assumed to be a quaternion. The quaternion is then normalized and converted to a rotation matrix.
            q = self.opt.variable(1, 4)
            self.q = q
            self.opt.subject_to(casadi.norm_2(q) == 1)
            # Convert the quaternion to a rotation matrix R0.
            casadi.MX.zeros((3, 3))
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

        self.ux = ux
        self.uy = uy
        self.uz = uz

        self.x0 = casadi.vertcat(p0, R0.flatten())
        self.x_stage = self.opt.variable(12, n_steps - 1)
        self.x = casadi.horzcat(self.x0, self.x_stage)

        self.s_val = np.linspace(0, self.l[-1], self.n_steps)
    
    @property
    def method( self ):
        return self.__method

    @method.setter
    def method( self, value ):
        if value not in ["rk1", "rk2", "euler", "rk3", "simpson", "heun", "rk4", "3/8"]:
            raise Exception(
                f"The given solver method is not valid or available. Please use one of the following solvers: {['rk1', 'rk2', 'euler', 'rk3', 'simpson', 'heun', 'rk4', '3/8']}"
                )
        self.__method = value

    @property
    def ux( self ):
        return self.__ux

    @ux.setter
    def ux( self, value ):
        if issubclass(type(value), ParameterFunctionCasadi):
            self.__ux = value
        else:
            self.__ux = Polynomial3Casadi(
                self.opt, self.n, value, optimize=self.optimize[0], continuous=self.continuous_u[0], random_init=self.random_init[0]
                )

    @property
    def uy( self ):
        return self.__uy

    @uy.setter
    def uy( self, value ):
        if issubclass(type(value), ParameterFunctionCasadi):
            self.__uy = value
        else:
            self.__uy = Polynomial3Casadi(
                self.opt, self.n, value, optimize=self.optimize[1], continuous=self.continuous_u[1], random_init=self.random_init[1]
                )

    @property
    def uz( self ):
        return self.__uz

    @uz.setter
    def uz( self, value ):
        if issubclass(type(value), ParameterFunctionCasadi):
            self.__uz = value
        else:
            self.__uz = Polynomial3Casadi(
                self.opt, self.n, value, optimize=self.optimize[2], continuous=self.continuous_u[2], random_init=self.random_init[2]
                )

    def set_funs( self ):
        """
        Function that fills the curvature function handles. Unlike the PyTorch implementation, the Casadi implementation does not need to be called every
        time the curvature parameters are changed. Instead, the curvature functions are set once and then used for the integration.
        """
        for u in ["ux", "uy", "uz"]:
            if not hasattr(self, u):
                setattr(self, u, None)

        self.s_val = np.linspace(0, self.l[-1], self.n_steps)

        self.ux_funs = [self.ux.get_u_fun(0, 0, self.l[0])]
        self.uy_funs = [self.uy.get_u_fun(0, 0, self.l[0])]
        self.uz_funs = [self.uz.get_u_fun(0, 0, self.l[0])]

        for i in range(1, self.n):
            self.ux_funs.append(self.ux.get_u_fun(i, self.l[i - 1], self.l[i]))
            self.uy_funs.append(self.uy.get_u_fun(i, self.l[i - 1], self.l[i]))
            self.uz_funs.append(self.uz.get_u_fun(i, self.l[i - 1], self.l[i]))

    def add_u_constraint( self, ax, lower_bound=None, upper_bound=None ):
        """
        Function for adding bounds to the curvature parameters. The constraints are added to the Casadi optimization problem.

        Arguments
        ----------
        ax : str
            The axis for which the curvature parameters are constrained. Can be one of the following: "x", "y", "z".

        lower_bound : double, np.array, or casadi.MX
            The lower bound for the curvature parameters. If a scalar is given, the same bound is used for all curvature parameters. If instead a vector
            is given, the bound is set for each curvature parameter individually.

        upper_bound : double, np.array, or casadi.MX
            The upper bound for the curvature parameters. If a scalar is given, the same bound is used for all curvature parameters. If instead a vector
            is given, the bound is set for each curvature parameter individually.
        """
        if ax == "x":
            u = self.ux.u_p
        elif ax == "y":
            u = self.uy.u_p
        elif ax == "z":
            u = self.uz.u_p
        else:
            raise Exception(
                'The given ax is not known, please only provide one of the following axes to constrain: "x", "y", "z".'
                )

        if lower_bound is not None:
            if type(lower_bound) is int:
                self.opt.subject_to(lower_bound <= u[:])
            else:
                self.opt.subject_to(lower_bound[:] <= u[:])

        if upper_bound is not None:
            if type(upper_bound) is int:
                self.opt.subject_to(u[:] <= upper_bound)
            else:
                self.opt.subject_to(u[:] <= upper_bound[:])

    def u_hat( self, s ):
        """
            Function for transforming u_p to the necessary skew-symmstric matrix. idx_fun is the index of the curvature function that is active at the current
            arc-length.

            Arguments
            ----------
            s : double
                The arc-length at which the curvature function is evaluated.

            Returns
            ----------
            out_pos : 3-by-3 np.array or casadi.MX
                The skew-symmetric matrix corresponding to the curvature functions at the arc-length s.
        """
        if s <= self.l[-1]:
            active_tube_idx = np.where(self.l >= s)[0]
            idx_fun = np.min(active_tube_idx)
        else:
            idx_fun = self.n - 1
        curvature_mat = casadi.MX(np.zeros((3, 3)))
        ux, uy, uz = self.ux_funs[idx_fun](s), self.uy_funs[idx_fun](s), self.uz_funs[idx_fun](s)
        curvature_mat[2, 1] = ux
        curvature_mat[1, 0] = uz
        curvature_mat[0, 2] = uy
        return curvature_mat - curvature_mat.T

    def setup( self, s=None ):
        """
        Integrates the Serret-Frenet formulas using the selected method in the variable self.method and sets the continuity constraints for the
        full-discretization of the ODE.
        """
        
        for i in range(self.n_steps - 1):
            # Do normal integration step if no value of s lies between self.s_val[i] and self.s_val[i+1]
            self.opt.subject_to(
                casadi.cse(self.x[:, i+1] == discretizeODE(
                    self.ode, self.method, 12, self.s_val[i+1]-self.s_val[i], self.n, self.s_val[i], self.x[:, i]
                    )
                )
            )
            
            
    def integrate( self, s ):
        x = casadi.MX(12, s.shape[0])
        for i in range(s.shape[0]):
            # Find the index j of self.s_val, where self.s_val[j] <= s[i]
            j = np.where(self.s_val <= s[i])[0][-1]
            x[:, i] = discretizeODE(
                self.ode, self.method, 12, s[i]-self.s_val[j], self.n, self.s_val[j], self.x[:, j]
                )
        return casadi.cse(x)


    def ode( self, s, x ):
        """
        Right-hand side of the moving frame ODE.

        Arguments
        ----------
        s : float
            The current value of the arc-length.

        x : 12-by-1 casadi.MX
            The current state x = [p, R]. p is the position of the CTCR in world coordinates and R is the rotation matrix describing the orientation of the
            CTCR at the current arc-length.
        """
        R = x[3:].reshape((3, 3))

        dp = R[:, 2]
        dR = (R @ self.u_hat(s)).reshape((9, 1))
        return casadi.vertcat(dp, dR)
    
    def get_p(self):
        return self.x[:3,:]
    
    def collect_parameters( self ):
        params = [u.get_parameters()[:] for u in [self.ux, self.uy, self.uz] if u.optimize]
        self.function_parameters = casadi.vertcat(*params)

class CasadiCurveEstimator:
    r"""
        This class implements the Casadi-based curve estimator (CCE) for continuum robots.
        The CCE takes a parameterized model for the continuum robot - this can be a rotation
        frame, circular arcs, differentiable rendering model, ... - and optimizes for the
        torch.parameters using an Iterative Closest Point algorithm.

        Arguments
        ----------
        curve_model : nn.Module
            A 3D point where the CTCR exits the actuation unit in world
            coordinates.
        
        l : positive scalar
            List of the lengths up to which the corresponding polynomial is used. Is
            typically selected to be each NiTi tube's length outside of the actuation
            unit.

        camera_calibration_parameters : list of dictionaries
            A list of dictionaries containing the keys
            "A", "dist", "P", "R", and "T". The function "camera_folder_to_params" can
            automatically generate this list for you.

        n_steps : int
            Number of intermediate steps to evaluate the reconstructed ctcr at.
            If low, the computation is fast, but for very low values the convergence might
            be bad. If high, a better solution might be found, at cost of higher computation
            times.

        w : double
            Weighting of the two cost functions. 0 (only track distance of pixels to
            reconstruction) is typically a good value, but if parts of the ctcr are occluded,
            increasing w can help.

        dist_norm : int
            Order of the metric used to penalize distances. Higher values give more cost to big errors.
    """

    def __init__( self, curve_model, camera_calibration_parameters, l, w=0., n_steps=50, dist_norm=2 ):

        self.camera_calibration_parameters = camera_calibration_parameters
        self.__bb_pixel_coordinates = None
        self.curve_model = curve_model
        self.opt = self.curve_model.opt
        self.w = w

        self.n_steps = n_steps

        self.dist_norm = dist_norm
        self.last_integrate_parameters = None
        self.ode_solution = None
        
           
    def initial_solve( self ):
        """
        Do an initial solve of the optimization problem. This is necessary to compute the positions p in the computation of the losses
        pixel_loss and backbone_loss.
        """
        self.curve_model.set_funs()
        self.curve_model.setup()
        self.curve_model.collect_parameters()
        self._set_warmstart_ipopt(1e-15)
        self.opt.minimize(0)
        sol = self.opt.solve()
        self.opt.solver('ipopt')
        self.sol = sol
        

    def solve_3d( self, s, pts, lam_2=0 ):
        """
        Solve the optimization problem for the given 3D points pts at the estimated positions s. The optimization problem is solved using Casadi. The resulting
        curvature parameters can then be used to reconstruct the CTCR's backbone. The program first sets the curvature functions, followed by the construction
        of the continuity constraints for the Serret-Frenet formulas. The resulting ODE is integrated using the selected method in the variable self.method.

        Arguments
        ----------
        s : m-by-1 np.array
            The arc-lengths at which the 3D points p were estimated.

        pts : m-by-3 np.array
            The 3D points that were estimated at the arc-lengths s.

        Returns
        ----------
        ux : n-by-4 np.array
            Values of the x-curvature parameters for all x-curvature function.

        uy : n-by-4 np.array
            Values of the y-curvature parameters for all y-curvature function.

        uz : n-by-4 np.array
            Values of the z-curvature parameters for all z-curvature function.
        """
        self.curve_model.set_funs()
        self.curve_model.setup()
        self.curve_model.collect_parameters()
        x_out = self.curve_model.integrate(s)
        cost = casadi.sumsqr(x_out[:3,:] - pts)
        self.opt.minimize(cost+lam_2*(casadi.sumsqr(self.curve_model.function_parameters)))
        sol = self.opt.solve()
        self.sol = sol

        ux = self.curve_model.ux.get_value(sol)
        uy = self.curve_model.uy.get_value(sol)
        uz = self.curve_model.uz.get_value(sol)

        return ux, uy, uz

    def get_img_coordinates( self ):
        """
        Project the computed backbone points to all the images and return the image coordinates.
        """
        if self.__bb_pixel_coordinates is None:
            backbone_pts = self.curve_model.get_p()
            self.__bb_pixel_coordinates = []
            for i in range(len(self.camera_calibration_parameters)):
                self.__bb_pixel_coordinates.append(fromWorld2ImgCasadi(backbone_pts, **self.camera_calibration_parameters[i]).T)
        return self.__bb_pixel_coordinates

    def get_pixel_diff_idx( self, ctcr_img_coordinates, img_idx_data ):
        """
        Computes for each CTCR pixel the index of the closest reconstruction pixel.

        Arguments
        ----------
        ctcr_img_coordinates : m-by-2 int
            Coordinates of CTCR pixels in image coordinates.

        img_idx_data : m-by-1 int
            Vector containing the index from which image the corresponding img_coordinates entry was taken.
        """
        backbone_img_coordinates = [self.sol.value(i) for i in self.get_img_coordinates()]
        idx_min_dist = []
        ctcr_img_coordinates_sorted = []
        for i in np.unique(np.unique(img_idx_data)):
            ctcr_img_coordinates_i = ctcr_img_coordinates[img_idx_data == i, :]
            x_diffs = backbone_img_coordinates[i][:, 0].reshape((-1, 1)) - ctcr_img_coordinates_i[:, 0].reshape(1, -1)
            y_diffs = backbone_img_coordinates[i][:, 1].reshape((-1, 1)) - ctcr_img_coordinates_i[:, 1].reshape(1, -1)
            distances = (x_diffs ** self.dist_norm + y_diffs ** self.dist_norm) ** (1 / self.dist_norm)
            idx_min_dist.append(np.argmin(distances, 0))
            ctcr_img_coordinates_sorted.append(ctcr_img_coordinates_i)
        return idx_min_dist, ctcr_img_coordinates_sorted

    def pixel_diff( self, ctcr_img_coordinates, img_idx_data ):
        """
        Compute the distance of each CTCR pixel to the closest reconstruction point in image coordinates using casadi.

        Arguments
        ----------
        ctcr_img_coordinates : m-by-2 int
            The image coordinates of CTCR pixels.

        img_idx_data :m-by-1 int
            The index of the image a point was taken from.
       """
        curve_pixel_coordinates = self.get_img_coordinates()
        diffs = None
        idx_min_dist, ctcr_img_coordinates_sorted = self.get_pixel_diff_idx(ctcr_img_coordinates, img_idx_data)
        for i in np.unique(img_idx_data):
            diffs = curve_pixel_coordinates[i][idx_min_dist[i], :] - ctcr_img_coordinates_sorted[i] if diffs is None else casadi.vertcat(
                diffs, curve_pixel_coordinates[i][idx_min_dist[i], :] - ctcr_img_coordinates_sorted[i])
        return diffs

    def pixel_loss( self, ctcr_img_coordinates, img_idx_data ):
        """
            Computes the average distance of each CTCR pixel to the corresponding backbone point in casadi.
        """
        # Penalizes the minimum distance of each measurement point to the reconstructed curve.
        diffs = self.pixel_diff(ctcr_img_coordinates, img_idx_data)
        return casadi.sum1(casadi.sum2(diffs ** self.dist_norm) ** (1 / self.dist_norm)) / diffs.shape[0]

    def get_backbone_diff_idx( self, ctcr_img_coordinates, img_idx_data ):
        """
        Computes for each backbone point in image coordinates the index of the closest CTCR pixel .

        Arguments
        ----------
        ctcr_img_coordinates : m-by-2 int
            Coordinates of CTCR pixels in image coordinates.

        img_idx_data : m-by-1 int
            Vector containing the index from which image the corresponding img_coordinates entry was taken.
        """
        backbone_img_coordinates = self.sol.value(self.get_img_coordinates())
        idx_min_dist = []
        ctcr_img_coordinates_sorted = None
        for i in np.unique(np.unique(img_idx_data)):
            ctcr_img_coordinates_i = ctcr_img_coordinates[img_idx_data == i, :]
            x_diffs = backbone_img_coordinates[i][:, 0].reshape((-1, 1)) - ctcr_img_coordinates_i[:, 0].reshape(1, -1)
            y_diffs = backbone_img_coordinates[i][:, 1].reshape((-1, 1)) - ctcr_img_coordinates_i[:, 1].reshape(1, -1)
            distances = (x_diffs ** self.dist_norm + y_diffs ** self.dist_norm) ** (1 / self.dist_norm)
            idx_min_dist.append(np.argmin(distances, 1))
            ctcr_img_coordinates_sorted = [ctcr_img_coordinates_i] if ctcr_img_coordinates_sorted is None else ctcr_img_coordinates_sorted + [
                ctcr_img_coordinates_i]

        return idx_min_dist, ctcr_img_coordinates_sorted

    def backbone_diff( self, ctcr_img_coordinates, img_idx_data ):
        """
        Compute the distance of each CTCR pixel to the closest reconstruction point in image coordinates.

        Arguments
        ----------
        ctcr_img_coordinates : m-by-2 int
            The image coordinates of CTCR pixels.

        img_idx_data :m-by-1 int
            The index of the image a point was taken from.
       """
        curve_pixel_coordinates = self.get_img_coordinates()
        diffs = None
        idx_min_dist, ctcr_img_coordinates_sorted = self.get_backbone_diff_idx(ctcr_img_coordinates, img_idx_data)
        for i in np.unique(img_idx_data):
            diffs = curve_pixel_coordinates[i] - ctcr_img_coordinates_sorted[i][idx_min_dist[i], :] if diffs is None else torch.concatenate(
                (diffs, curve_pixel_coordinates[i] - ctcr_img_coordinates_sorted[i][idx_min_dist[i], :]), 0)
        return diffs

    def backbone_loss( self, img_coordinates, img_idx_data ):
        """
            Computes the average distance of each backbone point to the corresponding CTCR pixel.
        """
        # Penalizes the minimum distance of each measurement point to the reconstructed curve.
        return torch.linalg.vector_norm(self.backbone_diff(img_coordinates, img_idx_data), self.dist_norm, 1).mean()

    def loss( self, img_coordinates, img_idx_data ):
        """
        Combines the two cost as a weighted sum using w as the weight.

        Arguments
        ----------
        img_coordinates : m-by-2 np.array
            Tensor containing the image coordinates of CTCR pixels.

        img_idx_data :m-by-1 np.array
            Tensor containing the index of the image a point was taken from.
       """
        loss = 0
        if self.w < 1.:
            loss += self.pixel_loss(img_coordinates, img_idx_data) * (1 - self.w)
        if self.w > 0.:
            loss += self.backbone_loss(img_coordinates, img_idx_data) * self.w
        return casadi.cse(loss)

    def loss_3d_dist( self, s, pts ):
        """
        Computes the sum of the squared distances of the reconstructed 3D points to the estimated 3D points.

        Arguments
        ----------
        s : m-by-1 np.array or casadi.MX
            The arc-lengths at which the 3D points p were estimated.

        pts : m-by-3 np.array or casadi.MX
            The 3D points that were estimated at the arc-lengths s.
        """
        pt_idc_0 = np.argmin(np.abs(self.s_val.reshape(-1, 1) - s.reshape(1, -1)), 0)

        diff = self.x[:3, pt_idc_0] - pts
        return casadi.sumsqr(diff)

    def reset_bb_pixel_coordinates( self ):
        """
            Resets the backbone pixel coordinates to None, so that they are recomputed the next time they are needed. This is useful if the camera parameters
            change, but typically not necessary. Calling this function may increase computation time as the backbone pixel coordinates are recomputed,
            instead of reused. This results in a larger computational graph in Casadi than necessary.
        """
        self.__bb_pixel_coordinates = None
    
    def _warmstart_with_prev(self):
        """
        Initializes the optimization variables with values from the previous solution.

        This method is designed to provide a warm start for the optimization process by initializing the current optimization variables with values obtained 
        from the previous solution. It sets the initial values for both the decision variables (self.opt.x) and the Lagrange multipliers (self.opt.lam_g) based 
        on the last solution (self.sol). This approach can potentially lead to faster convergence in iterative optimization procedures.
        """
        cce_init = [self.sol.value(self.opt.x), self.sol.value(self.opt.lam_g)]
        self.opt.set_initial(self.opt.x, cce_init[0])
        self.opt.set_initial(self.opt.lam_g, cce_init[1]) 
    
    def icp_step(self, p, img_idx, lam_2=0):
        """
        Performs an iterative closest point (ICP) step to align 3D points to a model.
        
        This method executes an ICP step using given 3D points, image indices, and an optional regularization parameter. It starts by initializing with previous 
        state (_warmstart_with_prev). The cost function is computed based on the loss between provided points and model, with an optional L2 regularization term
        scaled by lam_2. The optimization problem is then solved, and the solution is stored.
        
        Arguments
        ----------
        p : m-by-3 np.array or casadi.MX
            The 3D coordinates of points to be aligned. Each row represents a point.
        
        img_idx : n-by-1 np.array
            Indices of the images corresponding to each 3D point.
        
        lam_2 : float, optional
            Regularization parameter for L2 regularization. A higher value increases the weight of regularization in the cost function. Default is 0, indicating no regularization.
        
        Returns
        -------
        None
            This method updates the state of the object but does not return any value. The solution to the optimization problem is stored within the object.
        """
        self._warmstart_with_prev()
        loss = self.loss(p, img_idx)
        if lam_2 > 0:
            cost = loss + lam_2*(casadi.sumsqr(self.function_parameters))
        else:
            cost = loss
        self.opt.minimize(cost)
        self.sol = self.opt.solve()
        return self.sol.value(loss)
    
    def _set_warmstart_ipopt(self, val, **kwargs):
        """
        Configures the IPOPT solver for warm starting with specified parameters.
        
        This method sets up the IPOPT solver with warm start parameters. If additional keyword arguments are provided, they are used to configure the solver. If
        a list is provided as 'val', the method sets various IPOPT warm start parameters using elements of this list. If 'val' is not a list, it is used for all
        warm start parameters. This setup can help in achieving faster convergence for the optimization problem by utilizing information from previous solutions.
        
        Arguments
        ----------
        val : list or float
            If a list, it should contain values for 'mu_init', 'warm_start_mult_bound_push', 'warm_start_slack_bound_push', 
            'warm_start_bound_push', 'warm_start_bound_frac', and 'warm_start_slack_bound_frac', in that order. If a float, 
            the same value is used for all these parameters.
        
        **kwargs : dict, optional
            Additional keyword arguments to pass to the IPOPT solver for customization.
        
        Returns
        -------
        None
            This method configures the IPOPT solver within the object but does not return any value.
        """

        default_params = {
            'print_level': 0, 
            'warm_start_init_point': 'yes'
        }
    
        if isinstance(val, list):
            warmstart_params = ["mu_init", "warm_start_mult_bound_push", "warm_start_slack_bound_push", 
                                "warm_start_bound_push", "warm_start_bound_frac", "warm_start_slack_bound_frac"]
            default_params.update({param: value for param, value in zip(warmstart_params, val)})
        else:
            default_params.update({param: val for param in ["mu_init", "warm_start_mult_bound_push", 
                                                            "warm_start_slack_bound_push", "warm_start_bound_push", 
                                                            "warm_start_bound_frac", "warm_start_slack_bound_frac"]})
    
        self.opt.solver('ipopt', {}, {**default_params, **kwargs})