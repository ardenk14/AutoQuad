import torch
import sys
sys.path.append('mppi_control')
from mppi import MPPI
import numpy as np

from quadFiles.quad import Quadcopter
from controller import Controller

class MppiController(Controller):
    """
    MPPI-based flight controller
    """

    def __init__(self, quad, num_samples=20000, horizon=1):
        super().__init__(quad)
        #self.env = env
        # def state_dot(self, t, state, cmd, wind):
        self.model = quad.forward_model #model
        #self.target_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        #self.target_state = np.zeros(21, dtype=np.float32)
        self.target_state = np.array([  0.,           0.,          -10.,          1.,           0.,
                                        0.,           0.,           0.,           0.,           0.,
                                        0.,           0.,           0.,         522.98471407,   0.,
                                        522.98471407,   0.,         522.98471407,   0.,         522.98471407,
                                        0.        ], dtype=np.float32)
        self.test_action = np.array([522.9847140, 522.9847140, 522.9847140, 522.9847140], dtype=np.float32)
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = 21 #env.observation_space.shape[0]
        u_min = torch.ones(4) * quad.params["minWmotor"]
        u_max = torch.ones(4) * quad.params["maxWmotor"]
        u_init = torch.ones(4) * quad.params["w_hover"] #+ 20
        noise_sigma = 10 * torch.eye(4)
        lambda_value = 0.1

        cost_function = self.free_flight_cost_function
        
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max,
                         u_init=u_init)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = self.model(state, action)
        return next_state

    def control(self, quad, sDes, Ts):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        """
        state = quad.state
        # Convert numpy array to a tensor
        state_tensor = torch.from_numpy(state)
        # Get mppi command
        action_tensor = self.mppi.command(state_tensor)
        # Convert returned action from a tensor to a numpy array
        action = action_tensor.detach().numpy()
        print("ACTION: ", action)
        
        self.w_cmd = action
        #return action

    def free_flight_cost_function(self, state, action):
        """
        Compute the state cost for MPPI on a setup without obstacles.
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, state_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        # TODO: Add cost for next action too far from last action and actions going outside bounds of commands
        # TODO: Allow for changing target
        cost = None

        #Q = torch.eye(state.shape[-1], device=state.device)
        #Q = torch.zeros((state.shape[-1], state.shape[-1]))
        #Q[0, 0] = 2000.
        #Q[1, 1] = 2000.
        #Q[3, 3] = 5000.
        #Q[4, 4] = 5000.
        #Q[5, 5] = 5000.
        #Q[6, 6] = 5000.
        #Q[2, 2] = 0.
        Q = torch.eye(action.shape[-1])

        # TODO: Convert quaternion in batch to rotation matrix and find offset between up vector and z straight up and add to cost
        
        
        cost = torch.diagonal((action - self.test_action) @ Q @ (action - self.test_action).t()) #torch.diagonal((state - self.target_state) @ Q @ (state - self.target_state).t())
        return cost