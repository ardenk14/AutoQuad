import torch
import numpy as np

spacing = 0.1

def free_flight_cost_function(state, action, prev_action=None):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    # TODO: Add cost for next action too far from last action and actions going outside bounds of commands
    # TODO: Allow for changing target
    target_pose = 0 #TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None

    Q = torch.eye(state.shape[-1], device=state.device)
    
    cost = torch.diagonal((state - target_pose) @ Q @ (state - target_pose).t())
    return cost

def obstacle_avoidance_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    # TODO: Allow for changing target
    target_pose = 0 #TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None

    Q = torch.eye(3)
    Q[2, 2] = 0.1
    cost = torch.diagonal((state - target_pose) @ Q @ (state - target_pose).t()) #+ 100 * collision_detection(state)

    return cost
