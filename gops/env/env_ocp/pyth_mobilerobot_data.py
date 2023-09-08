from typing import Any, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding
from copy import deepcopy

from gops.env.env_ocp.pyth_base_data import PythBaseEnv
 
gym.logger.setLevel(gym.logger.ERROR)


class PythMobilerobot(PythBaseEnv):
    def __init__(
        self, **kwargs: Any,
    ):
        self.n_obstacle = 1
        self.safe_margin = 0.15
        self.max_episode_steps = 200
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of robot state
            robot_high = np.array([4, 1, 2*np.pi/3, 0.3, 0], dtype=np.float32)
            robot_low = np.array([2, -1, np.pi/3, 0, 0], dtype=np.float32)

            # initial range of tracking error
            error_high = np.zeros(3, dtype=np.float32)
            error_low = np.zeros(3, dtype=np.float32)

            # initial range of obstacle
            obstacle_high = np.array([2, 8, -np.pi/3, 0.5, 0], dtype=np.float32)
            obstacle_low = np.array(
                [0, 0, -2*np.pi/3, 0.0, 0], dtype=np.float32
            )

            init_high = np.concatenate(
                [robot_high, error_high] + [obstacle_high] * self.n_obstacle
            )
            init_low = np.concatenate(
                [robot_low, error_low] + [obstacle_low] * self.n_obstacle
            )
            work_space = np.stack((init_low, init_high))
        super(PythMobilerobot, self).__init__(work_space=work_space, **kwargs)

        self.robot = Robot()
        self.obses = [Robot() for _ in range(self.n_obstacle)]
        self.dt = 0.2
        self.state_dim = 2 + 3 + self.n_obstacle * 5
        self.action_dim = 2
        self.use_constraint = kwargs.get("use_constraint", True)
        self.constraint_dim = self.n_obstacle

        # lb_state = np.array(
        #     [-30, -30, -np.pi, -1, -np.pi / 2]
        #     + [-30, -np.pi, -2]
        #     + [-30, -30, -np.pi, -1, -np.pi / 2] * self.n_obstacle
        # )
        # hb_state = np.array(
        #     [60, 30, np.pi, 1, np.pi / 2]
        #     + [30, np.pi, 2]
        #     + [30, 30, np.pi, 1, np.pi / 2] * self.n_obstacle
        # )
        lb_state = np.array(
            [-1, -np.pi / 2]
            + [-30, -np.pi, -2]
            + [-30, -30, -np.pi, -1, -np.pi / 2] * self.n_obstacle
        )
        hb_state = np.array(
            [1, np.pi / 2]
            + [30, np.pi, 2]
            + [30, 30, np.pi, 1, np.pi / 2] * self.n_obstacle
        )
        # lb_action = np.array([-0.4, -np.pi / 3])
        # hb_action = np.array([0.4, np.pi / 3])

        # v_delta_max=1.8
        # w_delta_max=0.8
        lb_action = np.array([-1.8*self.dt, -0.8*self.dt])
        hb_action = np.array([1.8*self.dt, 0.8*self.dt])

        self.action_space = spaces.Box(low=lb_action, high=hb_action)
        self.observation_space = spaces.Box(lb_state, hb_state)

        self.seed()
        self.state_absolute = np.zeros(3 + (self.n_obstacle + 1) * 5)
        self.obs = self.reset()

        self.steps = 0

    @property
    def additional_info(self):
        return {
            "state": {"shape": (3 + (self.n_obstacle + 1) * 5,), "dtype": np.float32}, "constraint": {"shape": (0,), "dtype": np.float32},
        }

    @property
    def state(self):
        return self.state_absolute.reshape(-1)

    def reset(self, init_state: list = None, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        if init_state is None:
            state = [self.sample_initial_state()]
        else:
            state = [init_state]
        state = np.array(state, dtype=np.float32)
        state[:, 5:8] = self.robot.tracking_error(state[:, :5])
        self.steps_beyond_done = None
        self.steps = 0
        self.state_absolute = state
        self.obs = self.get_observation(self.state_absolute)

        return self.obs.reshape(-1), {"state": self.state_absolute, "constraint": self.get_constraint()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, dict]:
        #  define your forward function here: the format is just like: state_next = f(state,action)
        action = action.reshape(1, -1)
        for i in range(1 + self.n_obstacle):
            if i == 0:
                robot_state = self.robot.f_xu(
                    self.state_absolute[:, :5], action.reshape(1, -1), self.dt, "ego"
                )
                tracking_error = self.robot.tracking_error(robot_state)
                state_next = np.concatenate((robot_state, tracking_error), 1)

            else:
                obs_state = self.robot.f_xu(
                    self.state_absolute[:, 3 + i * 5 : 3 + i * 5 + 5],
                    np.zeros_like(self.state_absolute[:, 3 + i * 5 + 3 : 3 + i * 5 + 5]),
                    self.dt,
                    "obs",
                )
                state_next = np.concatenate((state_next, obs_state), 1)

        self.state_absolute = state_next
        self.obs = self.get_observation(self.state_absolute)

        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        # r_tracking = (
        #     -1.4 * np.square(tracking_error[:, 0])
        #     - 1 * np.square(tracking_error[:, 1])
        #     - 16 * np.square(tracking_error[:, 2])
        # )
        # r_action = -0.2 * np.square(action[:, 0]) - 0.5 * np.square(action[:, 1])
        r_tracking = (
                - 8 * np.square(self.obs[:, 2])  #  目标横向误差
                - 4 * np.square(self.obs[:, 3])  #  目标朝向角度误差
                - 4.5 * np.square(self.obs[:, 4])#  目标速度误差
                - 0.5 * np.square(self.obs[:, 1])    #  omega惩罚
                )
        r_action = -0.5 * np.square(action[:, 0]) - 0.5 * np.square(action[:, 1])   #delta v和delta omega惩罚
        reward = r_tracking + r_action

        # define the constraint here
        constraint = self.get_constraint()
        dead = constraint > 0
        reward = reward - 400 * (dead * constraint).sum(-1)

        # define the ending condition here the format is just like isdone = l(next_state)
        isdone = self.get_done()

        self.steps += 1
        info = {"state": self.state_absolute, "constraint": constraint}
        return (
            self.obs.reshape(-1),
            float(reward),
            isdone,
            info,
        )
    
    def get_observation(self, state):
        ego_x, ego_y, ego_theta, ego_v, ego_w = state[:, :5].T
        error_lat, error_theta, error_v = state[:, 5:8].T
        observation = np.stack((ego_v, ego_w, error_lat, error_theta, error_v), 1)
        for i in range(self.n_obstacle):
            obs_x, obs_y, obs_theta, obs_v, obs_w = state[:, 8 + i * 5 : 13 + i * 5].T
            rela_lon, rela_lat, rela_head = self.robot.transfer_to_another_coord(obs_x, obs_y, obs_theta, ego_x, ego_y, ego_theta)
            rela_v = obs_v - ego_v
            rela_w = obs_w - ego_w
            observation = np.concatenate((observation, rela_lon[:,np.newaxis], rela_lat[:,np.newaxis], rela_head[:,np.newaxis], rela_v[:,np.newaxis], rela_w[:,np.newaxis]), 1)
        return observation

    def get_done(self) -> np.ndarray:
        done = self.state_absolute[:, 0] < -5 or self.state_absolute[:, 1] > 6 or self.state_absolute[:, 1] < -6
        for i in range(self.n_obstacle):
            crush = (
                (
                    (
                        (self.state_absolute[:, 8 + i * 5] - self.state_absolute[:, 0]) ** 2
                        + (self.state_absolute[:, 9 + i * 5] - self.state_absolute[:, 1]) ** 2
                    )
                )
                ** 0.5
                - (
                    self.robot.robot_params["radius"]
                    + self.obses[i].robot_params["radius"]
                )
                < 0
            )
            done = done or crush
        return done

    def get_constraint(self) -> np.ndarray:
        constraint = np.zeros((self.state_absolute.shape[0], self.n_obstacle))
        for i in range(self.n_obstacle):
            safe_dis = (
                self.robot.robot_params["radius"]
                + self.obses[i].robot_params["radius"]
                + self.safe_margin
            )
            constraint[:, i] = (
                safe_dis
                - (
                    (
                        (self.state_absolute[:, 8 + i * 5] - self.state_absolute[:, 0]) ** 2
                        + (self.state_absolute[:, 9 + i * 5] - self.state_absolute[:, 1]) ** 2
                    )
                )
                ** 0.5
            )
        # return constraint.reshape(-1)
        return constraint

    def render(self, mode: str = "human", n_window: int = 1):

        if not hasattr(self, "artists"):
            self.render_init(n_window)
        state = self.state_absolute
        r_rob = self.robot.robot_params["radius"]
        r_obs = self.obses[0].robot_params["radius"]

        def arrow_pos(state):
            x, y, theta = state[0], state[1], state[2]
            return [x, x + np.cos(theta) * r_rob], [y, y + np.sin(theta) * r_rob]

        for i in range(n_window):
            for j in range(n_window):
                idx = i * n_window + j
                circles, arrows = self.artists[idx]
                circles[0].center = state[idx, :2]
                arrows[0].set_data(arrow_pos(state[idx, :5]))
                for k in range(self.n_obstacle):
                    circles[k + 1].center = state[
                        idx, 3 + (k + 1) * 5 : 3 + (k + 1) * 5 + 2
                    ]
                    arrows[k + 1].set_data(
                        arrow_pos(state[idx, 3 + (k + 1) * 5 : 3 + (k + 1) * 5 + 5])
                    )
            plt.pause(0.02)

    def render_init(self, n_window: int = 1):

        fig, axs = plt.subplots(n_window, n_window, figsize=(9, 9))
        artists = []

        r_rob = self.robot.robot_params["radius"]
        r_obs = self.obses[0].robot_params["radius"]
        for i in range(n_window):
            for j in range(n_window):
                if n_window == 1:
                    ax = axs
                else:
                    ax = axs[i, j]
                ax.set_aspect(1)
                ax.set_ylim(-6, 6)
                ax.set_xlim(-6, 6)
                ax.set_xticks([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
                ax.set_yticks([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
                ax.grid(True)
                x1 = np.linspace(-3, 0, 1000)
                y1 = 3 * np.ones_like(x1)
                y2 = np.linspace(0, 3, 1000)
                x2 = np.sqrt(1-np.power(y2/3, 2)) * 3
                y3 = np.linspace(-3, 0, 1000)
                x3 = 3 * np.ones_like(y3)
                ax.plot(x1, y1, "k")
                ax.plot(x2, y2, "k")
                ax.plot(x3, y3, "k")
                circles = []
                arrows = []
                circles.append(plt.Circle([0, 0], r_rob, color="red", fill=False))
                arrows.append(ax.plot([], [], "red")[0])
                ax.add_artist(circles[-1])
                ax.add_artist(arrows[-1])
                for k in range(self.n_obstacle):
                    circles.append(plt.Circle([0, 0], r_obs, color="blue", fill=False))
                    ax.add_artist(circles[-1])

                    arrows.append(ax.plot([], [], "blue")[0])
                artists.append([circles, arrows])
        self.artists = artists
        plt.ion()

    def close(self):
        plt.close("all")


class Robot:
    def __init__(self):
        self.robot_params = dict(
            v_max=0.4,
            w_max=np.pi / 2,
            v_delta_max=1.8,
            w_delta_max=0.8,
            v_desired=0.3,
            radius=0.74 / 2,  # per second
        )
        self.path = ReferencePath()

    def f_xu(
        self, states: np.ndarray, actions: np.ndarray, T: float, type: str
    ) -> np.ndarray:
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

        # delta_v = np.clip(v_cmd - v, -v_delta_max * T, v_delta_max * T)
        # delta_w = np.clip(w_cmd - w, -w_delta_max * T, w_delta_max * T)
        delta_v, delta_w = actions.T
        v_cmd = (
            np.clip(v + delta_v, 0, v_max)
            + np.random.normal(0, stds[0], [states.shape[0]]) * 0.5
        )
        w_cmd = (
            np.clip(w + delta_w, -w_max, w_max)
            + np.random.normal(0, stds[1], [states.shape[0]]) * 0.5
        )
        next_theta = theta + T * w_cmd
        # next_theta should be in the range of [-pi, pi], for example, if next_theta=3/2 * pi, it should be normalized to -1/2*pi
        next_theta = self.deal_with_theta_rad(next_theta)

        next_state = [
            x + T * np.cos(theta) * v_cmd,
            y + T * np.sin(theta) * v_cmd,
            next_theta,
            v_cmd,
            w_cmd,
        ]

        return np.stack(next_state, 1)

    def tracking_error(self, x: np.ndarray) -> np.ndarray:
        ref_x, ref_y, ref_phi = self.path.compute_path_point(x[:, 0], x[:, 1])
        error_lon, error_lat, error_head = self.transfer_to_another_coord(x[:, 0], x[:, 1], x[:, 2], ref_x, ref_y, ref_phi)

        error_v = x[:, 3] - self.robot_params["v_desired"]
        tracking = np.concatenate(
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
        rotated_x = orig_x * np.cos(rotate_phi) + orig_y * np.sin(rotate_phi)
        rotated_y = -orig_x * np.sin(rotate_phi) + orig_y * np.cos(rotate_phi)
        rotated_phi = self.deal_with_theta_rad(orig_phi - rotate_phi)
        return rotated_x, rotated_y, rotated_phi
    
    def deal_with_theta_rad(self, theta: np.ndarray):
        return (theta + np.pi) % (2 * np.pi) - np.pi

class ReferencePath(object):
    def __init__(self):
        pass

    # def compute_path_y(self, x: np.ndarray) -> np.ndarray:
    #     y = 0 * np.sin(1 / 3 * x)
    #     return y

    # def compute_path_phi(self, x: np.ndarray) -> np.ndarray:
    #     deriv = 0 * np.cos(1 / 3 * x)
    #     return np.arctan(deriv)
    
    def compute_path_point(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        phi = np.arctan2(y, x)
        if x <= 0:
            x_ref = deepcopy(x)
            y_ref = 3 * np.ones_like(x)
            theta_ref = np.pi * np.ones_like(x)
        else:
            if y < 0:
                x_ref = 3 * np.ones_like(x)
                y_ref = deepcopy(y)
                theta_ref = np.pi / 2 * np.ones_like(x)
            else:
                x_ref = 3 * np.cos(phi)
                y_ref = 3 * np.sin(phi)
                theta_ref = phi + np.pi / 2
        return x_ref, y_ref, theta_ref


def env_creator(**kwargs: Any):
    """
    make env `pyth_mobilerobot`
    """
    return PythMobilerobot(**kwargs)

if __name__ == "__main__":
    env = env_creator()
    env.reset()
    for _ in range(100):
        # action = env.action_space.sample()
        action = np.array([0, 0], dtype=np.float32)
        s, r, d, _ = env.step(action)
        print(env.state)
        env.render()
        if d: env.reset()