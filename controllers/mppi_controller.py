import torch
import sys
sys.path.append('mppi_control')
from controllers.mppi import MPPI
import numpy as np

from quadFiles.quad import Quadcopter
from controllers.controller import Controller

class MppiController(Controller):
    """
    MPPI-based flight controller
    """

    def __init__(self, quad, num_samples=1000, horizon=100):
        super().__init__(quad)
        #self.env = env
        # def state_dot(self, t, state, cmd, wind):
        self.model = quad.forward_model #model
        #self.target_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        #self.target_state = np.zeros(21, dtype=np.float32)
        self.target_state = np.array([  0.,           0.,          -4.,          1.,           0.,
                                        0.,           0.,           0.,           0.,           0.,
                                        0.,           0.,           0.,         522.98471407,   0.,
                                        522.98471407,   0.,         522.98471407,   0.,         522.98471407,
                                        0.        ], dtype=np.float32)
        self.test_action = np.array([522.9847140, 522.9847140, 522.9847140, 522.9847140], dtype=np.float32)
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = 21 #env.observation_space.shape[0]
        # Front left, front right, back
        u_min = torch.ones(4) * quad.params["minWmotor"]
        u_max = torch.ones(4) * quad.params["maxWmotor"]
        u_init = torch.ones(4) * quad.params["w_hover"] #+ 20
        noise_sigma = 10 * torch.eye(4)
        #noise_sigma[1, 1] = 500
        lambda_value = 0.1
        self.u_init = u_init

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
        # Adjust noise sigma so that more paths going towards stability than not
        quat_state = self.quaternions_to_matrix(state[3:7])
        # x forward, y to the right, z down
        # Whichever direction z pointed in, increase opposite by some amount
        #print(quat_state)
        x_direction = -quat_state[0, 2]
        y_direction = -quat_state[1, 2]
        direction = np.array([x_direction, y_direction])
        fl = direction @ np.array([1, -1] / np.sqrt(2))
        fr = direction @ np.array([1, 1] / np.sqrt(2))
        br = direction @ np.array([-1, 1] / np.sqrt(2))
        bl = direction @ np.array([-1, -1] / np.sqrt(2))
        # x is to the front and y is to the right. 
        # Motors fl, fr, br, bl: [+,-], [+, +], [-, +], [-, -]
        direction_length = direction @ direction
        noise_sigma = torch.tensor([[1., fl*fr, fl*br, fl*bl],
                                   [fr*fl, 1., fr*br, fr*bl],
                                   [br*fl, br*fr, 1., br*bl],
                                   [bl*fl, bl*fr, bl*br, 1.]]).float()
        noise_sigma = noise_sigma*noise_sigma * 30 * (1 + direction_length)
        #print("NOISE SIGMA: ", noise_sigma)
        

        # TODO: Set mean bias the same way
        #control_bias = self.mppi.noise_mu
        #print("LAST: ", (np.sum(self.w_cmd) / 4))
        # TODO: Add the difference in thrust based on tilt
        print("ST: ", quat_state[:, 2])
        motor_spds = np.hstack([state[13], state[15], state[17], state[19]])
        angle = (np.mean(motor_spds)) * (quat_state[:, 2]/np.linalg.norm(quat_state[:, 2]) @ np.array([0, 0, 1]))
        print("up thrust: ", angle)
        # TODO: Up thrust added to end is good but need derivative to slow down props when stable
        # up_thrust - last up thrust / Ts push to zero by the time we are stable
        up_thrust =  2 + 522.98471407 - angle #(np.mean(self.w_cmd)) * (quat_state[:, 2]/np.linalg.norm(quat_state[:, 2]) @ np.array([0, 0, 1]))
        control_bias = 30 * torch.tensor([fl + fl * (fr+br+bl), fr + fr*(fl+br+bl), br + br*(fl+fr+bl), bl + bl * (fl+fr+br)]).float() + up_thrust
        print(self.free_flight_cost_function(torch.tensor([state]).float(), control_bias))

        #self.mppi.u_init = self.u_init + control_bias
        self.mppi.set_noise_dist(noise_sigma=noise_sigma)

        print("BIAS: ", self.mppi.noise_mu)

        # Convert numpy array to a tensor
        state_tensor = torch.from_numpy(state)
        # Get mppi command
        action_tensor = self.mppi.command(state_tensor)
        # Convert returned action from a tensor to a numpy array
        action = action_tensor.detach().numpy()
        print("ACTION: ", action)
        
        self.w_cmd = action # self.mppi.best_action[0, 0].detach().numpy() #
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
        Q = torch.zeros((state.shape[-1], state.shape[-1]))
        Q[0, 0] = 1.
        Q[1, 1] = 1.
        #Q[3, 3] = 5000.
        #Q[4, 4] = 5000.
        #Q[5, 5] = 5000.
        #Q[6, 6] = 5000.
        Q[2, 2] = 1. #5000.
        Q_act = torch.eye(action.shape[-1])

        # Get angle between target state up vector and prospective state up vector
        target_quat = self.quaternions_to_z(torch.tensor([self.target_state[3:7]]))
        quat_state = self.quaternions_to_z(state[:, 3:7])
        dot = torch.acos(quat_state @ target_quat.t())[:, 0]

        # TODO: Convert quaternion in batch to rotation matrix and find offset between up vector and z straight up and add to cost
        #for i in state[:, 2]:
        #    if i < 0:
        #        print("State: ", i)
        
        #action_cost = torch.diagonal((action - self.test_action) @ Q_act @ (action - self.test_action).t())
        state_cost = torch.diagonal((state - self.target_state) @ Q @ (state - self.target_state).t())
        #print("cOST: ", cst.shape)
        #print("STATE SHAPE: ", state.shape)
        #values, indices = torch.topk(state_cost, int(state.shape[-1] / 4))
        #state_cost[indices] = 0.1
        cost = state_cost #+ dot**2 #torch.diagonal((state - self.target_state) @ Q @ (state - self.target_state).t())
        return cost
    
    def quaternions_to_z(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[:, 0]
        q1 = Q[:, 1]
        q2 = Q[:, 2]
        q3 = Q[:, 3]
        
        # First row of the rotation matrix
        #r00 = 2 * (q0 * q0 + q1 * q1) - 1
        #r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        #r10 = 2 * (q1 * q2 + q0 * q3)
        #r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        #r20 = 2 * (q1 * q3 - q0 * q2)
        #r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        #rot_matrix = np.array([[r00, r01, r02],
        #                    [r10, r11, r12],
        #                    [r20, r21, r22]])
        z_vec = torch.vstack([r02, r12, r22]).t()                              
        return z_vec
    
    def quaternions_to_matrix(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
        #z_vec = torch.vstack([r02, r12, r22]).t()                              
        return rot_matrix