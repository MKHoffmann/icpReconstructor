from dataclasses import dataclass
from copy import deepcopy

import torch
from ..torch_reconstruction import (
    TorchMovingFrame, 
    PolynomialKTorch,
    Polynomial3Torch,
    TorchCurveEstimator)
from ..utils import (
    PixelDataset,
    generate_circle,
    fromWorld2Img)


@dataclass
class Module:
    """
        Dataclass representing a single module of the modular bending
        actuator.

        Attributes
        ----------
        L : float
            Length of the module.
        a : float
            Distance from the backbone to the chamber.
        r : float
            Radius of the chamber.
        N : int
            Number of chambers in the module.
    """

    L: float = 0.05
    a: float = 0.011547
    r: float = 0.0075

    N: int = 3


class ModularBendingActuator(TorchMovingFrame):
    """
        Class representing a modular bending actuator.

        Attributes
        ----------
        modules : list[Module]
            List of modules in the actuator. Default is a single module.
        base_position : torch.Tensor with shape (3,)
            Base position of the actuator. Default is [0, 0, 0].
        base_rotation : torch.Tensor with shape (1,)
            Base rotation of the actuator. Default is 0 degrees.
        slender : bool
            Flag indicating if the actuator is slender. Default is False.
        curvature_degree : int
            Degree of the curvature polynomial. Default is 3.
    """

    def __init__(
            self,
            modules: list[Module] | None = None,
            base_position: torch.Tensor | None = None,
            base_rotation: torch.Tensor | None = None,
            slender: bool = False,
            curvature_degree: int = 3):

        # Modules
        if modules is None:
            self.modules = [Module()]
        else:
            self.modules = modules

        # Number of modules
        N = len(self.modules)

        # Generate list of lengths
        l = torch.tensor([module.L for module in self.modules])

        # Initialize TorchMovingFrame
        super().__init__(l, rotation_method="rotm")

        # Settings
        self.slender = slender

        # Base Rotation
        if base_rotation is None:
            # Base rotation is set to 0 degrees
            self.base_rotation = torch.tensor([0.0])
        else:
            # Base rotation is set to the given value
            self.base_rotation = torch.deg2rad(base_rotation)

        # Base Position
        if base_position is None:
            # Base position is at the origin
            self.base_position = torch.nn.Parameter(torch.zeros((3,)))
        else:
            # Base position is set to the given value
            self.base_position = torch.nn.Parameter(base_position)

        # Update the initial state of the moving frame
        self.update_initial_state()

        # Curvature (Polynomial)
        ux_p = torch.full((N, curvature_degree + 1), 0.0)
        uy_p = torch.full((N, curvature_degree + 1), 0.0)
        
        # NOTE: Override the curvature function for a cubic polynomial
        if curvature_degree != 3:
            self.ux = PolynomialKTorch(N, curvature_degree, u_p=ux_p)
            self.uy = PolynomialKTorch(N, curvature_degree, u_p=uy_p)
        else:
            self.ux = Polynomial3Torch(N, u_p=ux_p)
            self.uy = Polynomial3Torch(N, u_p=uy_p)

        # Torsion is set to constant zero and not optimized
        self.uz = PolynomialKTorch(N, 0, optimize=False)

        # Elongation (Constant)
        # NOTE: Initial stretch ratio is set to 1.0
        self.la = PolynomialKTorch(N, 0, u_p=torch.ones((N, 1))) 
        
        # Update to use the ode in this class
        self.ode = self.__rotm_ode

        # Set curvature and elongation functions
        self.set_curvature_functions()
        self.set_elongation_functions()

    def set_curvature_functions(
            self) -> None:
        """
            Set the curvature functions.
        """

        # Set the curvature functions for the TorchMovingFrame
        self.set_funs()

    def set_elongation_functions(
            self) -> None:
        """
            Set the elongation functions.
        """      

        self.la_funs = [self.la.get_u_fun(0, 0, self.l[0])]

        for i in range(1, self.n):
            self.la_funs.append(self.la.get_u_fun(i, self.l[i - 1], self.l[i]))

    def solve_backbone(
            self,
            s_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Solve the backbone of the actuator.

            Arguments
            ---------
            s_values : torch.Tensor with shape (N,)
                Values of the arclength where to evaluate the backbone.

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                Tuple containing the position and rotation of the 
                backbone.
        """

        # Solve ODE
        ode_solution = self.integrate(s_values)

        # Position and Rotation
        p_values = ode_solution[:, :3]
        R_values = ode_solution[:, 3:].reshape(-1, 3, 3)

        return p_values, R_values

    def from_actuator_space_to_world_space(
            self,
            points: torch.Tensor) -> torch.Tensor:
        """
            Transform points from actuator space to world space.

            Arguments
            ---------
            points : torch.Tensor with shape (N, 3)
                Points to transform.
            
            Returns
            -------
            torch.Tensor with shape (N, 3)
                Transformed points in world space.
        """

        # NOTE
        #   x_World =  y_SoftRobot
        #   y_World = -z_SoftRobot
        #   z_World = -x_SoftRobot 

        points = points[:, [1, 2, 0]]
        points[:, [1, 2]] *= -1

        points += self.base_position

        return points

    def generate_estimate_points(
            self,
            s_values: torch.Tensor,
            p_values: torch.Tensor,
            R_values: torch.Tensor) -> torch.Tensor:
        """
            Generate estimate points along the backbone.

            Arguments
            ---------
            s_values : torch.Tensor with shape (N,)
                Values of the arclength where to evaluate the backbone.
            p_values : torch.Tensor with shape (N, 3)
                Position of the backbone.
            R_values : torch.Tensor with shape (N, 3, 3)
                Rotation of the backbone.
            
            Returns
            -------
            torch.Tensor with shape (M, 3)
                Estimate points along the backbone.
        """

        estimate_points = []

        for s, p, R in zip(s_values, p_values, R_values):
            # Compute module indices based on s
            l_cumulative = torch.cumsum(self.l, dim=0)

            module_index = torch.argmax((s < l_cumulative).int())
            
            # Module 
            module = self.modules[module_index]

            # Inplane chamber center
            c = torch.from_numpy(generate_circle(module.N))

            # If slender, we only need to generate estimate points on 
            # the centerline
            if not self.slender:
                # Generate shell points for each chamber
                for j in range(module.N):
                    # Calculate inplane position of chamber related to 
                    # backbone

                    # Center point of chamber
                    p_c = p + (c[j, 0] * R[:, 0] + c[j, 1] * R[:, 1])*module.a

                    estimate_points.append(p_c)
            else:
                # Generate marker points on the centerline
                estimate_points.append(p)

        # Stack chamber backbone points in tensor
        estimate_points = torch.stack(estimate_points)

        # Transform the estimate points to world space
        return self.from_actuator_space_to_world_space(estimate_points)

    def evaluate_la(
            self,
            s: float) -> float:
        """
            Evaluate the elongation function at the given arclength.

            Arguments
            ---------
            s : float
                Arclength where to evaluate the elongation function.
            
            Returns
            -------
            float
                Value of the elongation function at the given arclength.
        """

        if s <= self.l[-1]:
            # s is inside
            module_index = torch.min(torch.where(self.l >= s)[0])
        else:
            # s is outside -> Use last module
            module_index = self.n - 1

        return self.la_funs[module_index](s)

    def update_initial_state(
            self) -> None:
        """
            Update the initial state of the actuator.
        """

        self.R0 = torch.tensor([
            [  1.0, -1.0,  0.0],
            [  1.0,  1.0,  0.0],
            [  0.0,  0.0,  1.0]])
        
        self.R0[0, 0] *= torch.cos(self.base_rotation)
        self.R0[0, 1] *= torch.sin(self.base_rotation)
        self.R0[1, 0] *= torch.sin(self.base_rotation)
        self.R0[1, 1] *= torch.cos(self.base_rotation)

        # NOTE: Explicitly set p0 to zero in soft robot frame. We will 
        #       set the initial base position of the soft robot later in
        #       world frame.
        self.p0 = torch.tensor([0.0, 0.0, 0.0])
        self.x0 = torch.concatenate((self.p0.flatten(), self.R0.flatten()), 0)

    def forward(
            self, 
            s_values: torch.Tensor | None = None) -> torch.Tensor:
        """
            Forward pass of the actuator. Solves the backbone and 
            generates estimate points along the backbone.

            Arguments
            ---------
            s_values : torch.Tensor with shape (N,)
                Values of the arclength where to evaluate the backbone.
            
            Returns
            -------
            torch.Tensor with shape (M, 3)
                Generated estimate points along the backbone.
        """
    
        # Recompute initial state
        self.update_initial_state()

        # Solve backbone
        p_values, R_values = self.solve_backbone(s_values)

        # Generate estimate points along the backbone
        return self.generate_estimate_points(s_values, p_values, R_values)

    def __rotm_ode(
            self,
            s: torch.Tensor,
            x: torch.Tensor) -> torch.Tensor:
        """
            Ordinary differential equation for the actuator using a 
            rotation matrix.

            Arguments
            ---------
            s : torch.Tensor
                Arclength where to evaluate the ode.
            x : torch.Tensor
                State of the ode

            Returns
            -------
            torch.Tensor with shape (12,)
                Derivative of the state.
        """
    
        R = x[3:].reshape((3, 3))

        dp = self.evaluate_la(s) * R[:, 2].flatten()
        dR = (R @ self.u_hat(s)).flatten()
        return torch.concat((dp, dR), 0)