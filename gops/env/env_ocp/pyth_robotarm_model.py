from typing import Tuple, Union
import torch
import numpy as np
from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class Dynamics(object):
    def __init__(self):
        max_torque = 100  # Nm
        lb_action = [-max_torque, -max_torque]
        hb_action = [max_torque, max_torque]
        self.action_upper_bound = torch.tensor(hb_action, dtype=torch.float32)
        self.action_lower_bound = torch.tensor(lb_action, dtype=torch.float32)
        self.action_center = (self.action_upper_bound + self.action_lower_bound) / 2
        self.action_half_range = (self.action_upper_bound - self.action_lower_bound) / 2

        # Define some physical parameters of the quadrotor
        self.L1 = 1
        self.m1 = 1
        self.L2 = 1
        self.m2 = 1
        self.I1 = 0.5
        self.I2 = 0.5

        self.c1 = (self.I1 + self.I2 + self.m1 * self.L1 ** 2 + self.m2 * (self.L1 ** 2 + self.L2 ** 2))
        self.c2 = self.m2 * self.L1 * self.L2
        self.c3 = self.I2

        self.g = 9.81  # gravity coefficient

        # control goal
        self.X_goal = [0.0, 0.0, 0.0, 0.0]

    def inverse_normalize_action(self, actions: torch.Tensor):
        actions = actions * self.action_half_range + self.action_center
        return actions
    
    def f_xu(self, states, actions, dt):
        # actions should be inputted in normalized form
        actions = self.inverse_normalize_action(actions)

        q1, dq1, q2, dq2 = states.T.unbind()
        torque1, torque2 = actions.T.unbind()

        ddq2 = (torque2 + (dq1 ** 2 * self.L1 + dq2 ** 2 * self.L2) * self.m2 * torch.sin(q2) -
            self.c2 * torch.sin(q2) * dq1 ** 2 - self.g * self.m2 * torch.sin(q1 + q2)) / self.c3
        ddq1 = (torque1 - dq2 ** 2 * self.L1 * self.m2 * torch.sin(q2) - self.g * (self.m1 + self.m2) * torch.sin(q1) +
            dq2 ** 2 * self.L2 * self.m2 * torch.sin(q2) + self.c2 * torch.cos(q2) * ddq2) / self.c1

        next_q1 = q1 + dt * dq1
        next_dq1 = dq1 + dt * ddq1
        next_q2 = q2 + dt * dq2
        next_dq2 = dq2 + dt * ddq2
        next_q1 = next_q1.reshape(-1, 1)
        next_dq1 = next_dq1.reshape(-1, 1)
        next_q2 = next_q2.reshape(-1, 1)
        next_dq2 = next_dq2.reshape(-1, 1)
        next_states = torch.cat([next_q1, next_dq1, next_q2, next_dq2], dim=1)
        return next_states

    def compute_rewards(self, states, actions):  # obses and actions are tensors
        actions = self.inverse_normalize_action(actions.squeeze(-1))

        q1, dq1, q2, dq2 = states.T.unbind()
        # TODO: Q matrix
        # cost for q1
        cost_q1 = 1 * torch.square(q1 - self.X_goal[0])
        # cost for dq1
        cost_dq1 = 0.5 * torch.square(dq1 - self.X_goal[1])
        # cost for q2
        cost_q2 = 1 * torch.square(q2 - self.X_goal[2])
        # cost for dq2
        cost_dq2 = 0.5 * torch.square(dq2 - self.X_goal[3])
        # cost for u
        act_penalty = 0.1 * torch.sum(torch.square(actions), dim=-1)
        # TODO: reward composition
        rewards = - cost_q1 - cost_dq1 - cost_q2 - cost_dq2 - act_penalty

        return rewards

    def get_done(self, states):
        q1, dq1, q2, dq2 = states.T.unbind()
        # low=np.array(lb_action, dtype=np.float32)
        d1 = torch.abs(q1) >= np.pi
        d2 = torch.abs(dq1) >= 10
        d3 = torch.abs(q2) >= np.pi
        d4 = torch.abs(dq2) >= 10
        done = d1 + d2 + d3 + d4
        done = done >= 1
        return done


class PythRobotArm(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None):
        obs_dim = 4
        action_dim = 2
        dt = 0.01
        self.discrete_num = 5
        lb_state = [-np.inf] * obs_dim
        hb_state = [np.inf] * obs_dim
        lb_action = [-1.0, -1.0]
        hb_action = [1.0, 1.0]
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            dt=dt,
            obs_lower_bound=lb_state,
            obs_upper_bound=hb_state,
            action_lower_bound=lb_action,
            action_upper_bound=hb_action,
            device=device,
        )
        # define your custom parameters here

        self.dynamics = Dynamics()

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs = obs
        for _ in range(self.discrete_num):
            next_obs = self.dynamics.f_xu(
                obs, action, self.dt / self.discrete_num
            )
            obs = next_obs
        reward = self.dynamics.compute_rewards(next_obs, action).reshape(-1)
        # done = torch.full([obs.size()[0]], False, dtype=torch.bool, device=self.device)
        done = self.dynamics.get_done(next_obs).reshape(-1)
        info = {"constraint": None}
        return next_obs, reward, done, info


def env_model_creator(**kwargs):
    """
    make env model `pyth_robotarm`
    """
    return PythRobotArm(kwargs["device"])
