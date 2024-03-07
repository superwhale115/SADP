import warnings

import numpy as np
import torch

from gops.utils.gops_typing import InfoDict


class GymDemocontiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        you need to define parameters here
        """
        # define your custom parameters here

        # define common parameters here
        self.state_dim = None
        self.action_dim = None
        lb_state = None
        hb_state = None
        lb_action = None
        hb_action = None
        self.dt = None  # seconds between state updates

        # do not change the following section
        self.lb_state = torch.tensor(lb_state, dtype=torch.float32)
        self.hb_state = torch.tensor(hb_state, dtype=torch.float32)
        self.lb_action = torch.tensor(lb_action, dtype=torch.float32)
        self.hb_action = torch.tensor(hb_action, dtype=torch.float32)

    def forward(self, state: torch.Tensor, action: torch.Tensor, info: InfoDict, beyond_done=None):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :param info: datatype: InfoDict, any useful information for debug or training, including constraint,
                     adversary action, etc
        :param beyond_done: flag indicate the state is already done which means it will not be calculated by the model
        :return:
                next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                             the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[batch_size,]
                isdone:   datatype:torch.Tensor, shape:[batch_size,]
                          flag done will be set to true when the model reaches the max_iteration or the next state
                          satisfies ending condition

                info: datatype: InfoDict, any useful information for debug or training, including constraint,
                      adversary action, etc
        """
        warning_msg = "action out of action space!"
        if not ((action <= self.hb_action).all() and (action >= self.lb_action).all()):
            warnings.warn(warning_msg)
            action = clip_by_tensor(action, self.lb_action, self.hb_action)
        # If the state is same as observation in your environment, you can delete the following line
            state = info["state"]

        warning_msg = "state out of state space!"
        if not ((state <= self.hb_state).all() and (state >= self.lb_state).all()):
            warnings.warn(warning_msg)
            state = clip_by_tensor(state, self.lb_state, self.hb_state)

        #  define your forward function here: the format is just like: state_next = f(state,action)
        state_next = (1 + action.sum(-1)) * state

        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = torch.full([state.size()[0]], False, dtype=torch.bool)

        ############################################################################################

        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = (state_next - state).sum(dim=-1)

        ############################################################################################
        if beyond_done is None:
            beyond_done = torch.full([state.size()[0]], False, dtype=torch.bool)

        beyond_done = beyond_done.bool()
        mask = isdone | beyond_done
        mask = torch.unsqueeze(mask, -1)
        state_next = ~mask * state_next + mask * state
        reward = ~(beyond_done) * reward
        # the content in info is determined by your environment, if your environment don't need any extra infomation,
        # you can set info ={}
        info={"state":state_next,"constraint": 0}
        return state_next, reward, mask.squeeze(), info

    def forward_n_step(self, func, n, state: torch.Tensor):
        reward = torch.zeros(size=[state.size()[0], n])
        isdone = state.numpy() <= self.hb_state | state.numpy() >= self.lb_state
        if np.sum(isdone) > 0:
            warning_msg = "state out of state space!"
            warnings.warn(warning_msg)
        isdone = torch.from_numpy(isdone)
        for step in range(n):
            action = func(state)
            state_next, reward[:, step], isdone, constraint = self.forward(state, action, isdone)
            state = state_next


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result


if __name__ == "__main__":
    env = GymDemocontiModel()
