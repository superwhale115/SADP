from typing import Any, Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict
from copy import deepcopy

def env_model_creator(**kwargs):
    return PythMobilerobotModel(kwargs["device"])


class PythMobilerobotModel(PythBaseModel):
    def __init__(
        self, device: Union[torch.device, str, None] = None, **kwargs: Any,
    ):
        self.n_obstacle = 1
        self.safe_margin = 0.15
        self.robot = Robot()
        self.obses = [Robot() for _ in range(self.n_obstacle)]

        # define common parameters here
        self.dt = 0.2
        self.state_dim = 2 + 3 + self.n_obstacle * 5
        self.action_dim = 2
        lb_state = (
            [-1, -np.pi / 2]
            + [-30, -np.pi, -2]
            + [-30, -30, -np.pi, -1, -np.pi / 2] * self.n_obstacle
        )
        hb_state = (
            [1, np.pi / 2]
            + [30, np.pi, 2]
            + [30, 30, np.pi, 1, np.pi / 2] * self.n_obstacle
        )
        lb_action = [-1.8*self.dt, -0.8*self.dt]
        hb_action = [1.8*self.dt, 0.8*self.dt]

        super().__init__(
            obs_dim=len(lb_state),
            action_dim=len(lb_action),
            dt=self.dt,
            obs_lower_bound=lb_state,
            obs_upper_bound=hb_state,
            action_lower_bound=lb_action,
            action_upper_bound=hb_action,
            device=device,
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        state = info["state"]
        #  define your forward function here: the format is just like: state_next = f(state,action)
        veh2vehdist = torch.zeros(state.shape[0], self.n_obstacle)
        for i in range(1 + self.n_obstacle):
            if i == 0:
                robot_state = self.robot.f_xu(state[:, :5], action, self.dt, "ego")
                tracking_error = self.robot.tracking_error(robot_state)
                state_next = torch.cat((robot_state, tracking_error), 1)

            else:
                obs_state = self.robot.f_xu(
                    state[:, 3 + i * 5 : 3 + i * 5 + 5],
                    torch.zeros_like(state[:, 3 + i * 5 + 3 : 3 + i * 5 + 5]),
                    self.dt,
                    "obs",
                )
                state_next = torch.cat((state_next, obs_state), 1)

                safe_dis = (
                    self.robot.robot_params["radius"]
                    + self.obses[i - 1].robot_params["radius"]
                    + self.safe_margin
                )  # 0.35
                veh2vehdist[:, i - 1] = safe_dis - (
                    torch.sqrt(
                        torch.square(state_next[:, 3 + i * 5] - state_next[:, 0])
                        + torch.square(state_next[:, 3 + i * 5 + 1] - state_next[:, 1])
                    )
                )

        obs_next = self.get_observation(state_next)
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        r_tracking = (
            -8 * torch.square(obs_next[:, 2])
            - 4 * torch.square(obs_next[:, 3])
            - 4.5 * torch.square(obs_next[:, 4])
            - 0.5 * torch.square(obs_next[:, 1])
        )
        r_action = -0.5 * torch.square(action[:, 0]) - 0.5 * torch.square(action[:, 1])
        reward = r_tracking + r_action

        # define the constraint funtion
        constraint = veh2vehdist
        dead = veh2vehdist > 0
        reward = reward - 400 * (dead * constraint).sum(-1)
        info = {"state": state_next, "constraint": constraint}
        # define the ending condition here the format is just like isdone = l(next_state)
        isdone = self.get_done(state_next, veh2vehdist)

        return obs_next, reward, isdone, info

    def get_observation(self, state):
        ego_x, ego_y, ego_theta, ego_v, ego_w = state[:, :5].T.unbind()
        error_lat, error_theta, error_v = state[:, 5:8].T.unbind()
        observation = torch.stack((ego_v, ego_w, error_lat, error_theta, error_v), 1)
        for i in range(self.n_obstacle):
            obs_x, obs_y, obs_theta, obs_v, obs_w = state[:, 8 + i * 5 : 13 + i * 5].T.unbind()
            rela_lon, rela_lat, rela_head = self.robot.transfer_to_another_coord(obs_x, obs_y, obs_theta, ego_x, ego_y, ego_theta)
            rela_v = obs_v - ego_v
            rela_w = obs_w - ego_w
            observation = torch.cat((observation, rela_lon.unsqueeze(1), rela_lat.unsqueeze(1), rela_head.unsqueeze(1), rela_v.unsqueeze(1), rela_w.unsqueeze(1)), 1)
        return observation
    
    def get_done(self, state: torch.Tensor, veh2vehdist: torch.Tensor) -> torch.Tensor:
        done = torch.logical_or(state[:, 0] < -5, torch.abs(state[:, 1]) > 6)
        for i in range(self.n_obstacle):
            crush = veh2vehdist[:, i] > self.safe_margin
            done = torch.logical_or(done, crush)
        return done


class Robot:
    def __init__(self):
        self.robot_params = dict(
            v_max=0.4,
            w_max=np.pi / 2,
            v_delta_max=1.8,
            w_delta_max=0.8,
            v_desired=0.3,
            radius=0.74 / 2,
        )
        self.path = ReferencePath()

    def f_xu(
        self, states: torch.Tensor, actions: torch.Tensor, T: float, type: str
    ) -> torch.Tensor:
        v_delta_max = self.robot_params["v_delta_max"]
        v_max = self.robot_params["v_max"]
        w_max = self.robot_params["w_max"]
        w_delta_max = self.robot_params["w_delta_max"]
        std_type = {
            "ego": [0.0, 0.0],
            "obs": [0.001, 0.001],
            "none": [0, 0],
            "explore": [0.001, 0.001],
        }
        stds = std_type[type]

        x, y, theta, v, w = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
        )
        # v_cmd, w_cmd = actions[:, 0], actions[:, 1]

        # delta_v = torch.clamp(v_cmd - v, -v_delta_max * T, v_delta_max * T)
        # delta_w = torch.clamp(w_cmd - w, -w_delta_max * T, w_delta_max * T)
        delta_v, delta_w = actions.T.unbind()
        v_cmd = (
            torch.clamp(v + delta_v, 0, v_max)
            + torch.Tensor(np.random.normal(0, stds[0], [states.shape[0]])) * 0.5
        )
        w_cmd = (
            torch.clamp(w + delta_w, -w_max, w_max)
            + torch.Tensor(np.random.normal(0, stds[1], [states.shape[0]])) * 0.5
        )
        next_theta = theta + T * w_cmd
        # next_theta should be in the range of [-pi, pi], for example, if next_theta=3/2 * pi, it should be normalized to -1/2*pi
        next_theta = self.deal_with_theta_rad(next_theta)

        next_state = [
            x + T * torch.cos(theta) * v_cmd,
            y + T * torch.sin(theta) * v_cmd,
            next_theta,
            v_cmd,
            w_cmd,
        ]

        return torch.stack(next_state, 1)

    def tracking_error(self, x: torch.Tensor) -> torch.Tensor:
        ref_x, ref_y, ref_phi = self.path.compute_path_point(x[:, 0], x[:, 1])
        error_lon, error_lat, error_head = self.transfer_to_another_coord(x[:, 0], x[:, 1], x[:, 2], ref_x, ref_y, ref_phi)


        error_v = x[:, 3] - self.robot_params["v_desired"]
        tracking = torch.cat(
            (
                error_lat.reshape(-1, 1),
                error_head.reshape(-1, 1),
                error_v.reshape(-1, 1),
            ),
            1,
        )
        return tracking

    def transfer_to_another_coord(self, x, y, theta, ego_x, ego_y, ego_theta):
        shift_x, shift_y = self.shift(x, y, ego_x, ego_y)
        x_ego_coord, y_ego_coord, phi_ego_coord = self.rotate(shift_x, shift_y, theta, ego_theta)
        return x_ego_coord, y_ego_coord, phi_ego_coord

    def shift(self, orig_x, orig_y, shift_x, shift_y):
        shifted_x = orig_x - shift_x
        shifted_y = orig_y - shift_y
        return shifted_x, shifted_y

    def rotate(self, orig_x, orig_y, orig_phi, rotate_phi):
        rotated_x = orig_x * torch.cos(rotate_phi) + orig_y * torch.sin(rotate_phi)
        rotated_y = -orig_x * torch.sin(rotate_phi) + orig_y * torch.cos(rotate_phi)
        rotated_phi = self.deal_with_theta_rad(orig_phi - rotate_phi)
        return rotated_x, rotated_y, rotated_phi
    
    def deal_with_theta_rad(self, theta: torch.Tensor):
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

class ReferencePath(object):
    def __init__(self):
        pass

    # def compute_path_y(self, x: torch.Tensor) -> torch.Tensor:
    #     y = 0 * torch.sin(1 / 3 * x)
    #     return y

    # def compute_path_phi(self, x: torch.Tensor) -> torch.Tensor:
    #     deriv = 0 * torch.cos(1 / 3 * x)
    #     return torch.arctan(deriv)
    
    def compute_path_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_bool = (x <= 0)
        y_bool = (y <= 0)
        phi = torch.atan2(y, x)

        x_ref = x_bool * x \
            + (~x_bool) * y_bool * 3 * torch.ones_like(x) \
            + (~x_bool) * (~y_bool) * 3 * torch.cos(phi)
        
        y_ref = x_bool * 3 * torch.ones_like(x) \
            + (~x_bool) * y_bool * y \
            + (~x_bool) * (~y_bool) * 3 * torch.sin(phi)
        
        theta_ref = x_bool * torch.pi * torch.ones_like(x) \
            + (~x_bool) * y_bool * torch.pi / 2 * torch.ones_like(x) \
            + (~x_bool) * (~y_bool) * (phi + torch.pi / 2)

        return x_ref.detach(), y_ref.detach(), theta_ref.detach()