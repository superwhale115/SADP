__all__ = ["INFGDHP"]

from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
import time

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        value_func_type = kwargs["value_func_type"]
        policy_func_type = kwargs["policy_func_type"]

        v_args = get_apprfunc_dict("value", value_func_type, **kwargs)
        policy_args = get_apprfunc_dict("policy", policy_func_type, **kwargs)

        self.v = create_apprfunc(**v_args)
        self.policy = create_apprfunc(**policy_args)

        self.v_target = deepcopy(self.v)
        self.policy_target = deepcopy(self.policy)

        for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )  #
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs["value_learning_rate"])

        self.net_dict = {"v": self.v, "policy": self.policy}
        self.target_net_dict = {"v": self.v_target, "policy": self.policy_target}
        self.optimizer_dict = {"v": self.v_optimizer, "policy": self.policy_optimizer}

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class INFGDHP(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.gamma = 0.99
        self.tau = 0.005
        self.forward_step = 10
        self.reward_scale = 1
        self.tb_info = dict()
        self.weight_loss_v = 0.2
        self.adapt_weights = False
        self.weights_saturate_iter = 1
        self.equilibrium = None

    @property
    def adjustable_parameters(self):
        para_tuple = ("gamma", "tau", "pev_step", "pim_step", "forward_step", "reward_scale", "weight_loss_v", "adapt_weights", "weights_saturate_iter", "equilibrium")
        return para_tuple

    def local_update(self, data: dict, iteration: int) -> dict:
        update_list = self.__compute_gradient(data, iteration)
        self.__update(update_list)
        return self.tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        update_list = self.__compute_gradient(data, iteration)
        update_info = dict()
        for net_name in update_list:
            update_info[net_name] = [p.grad for p in self.networks.net_dict[net_name].parameters()]
        return self.tb_info, update_info

    def remote_update(self, update_info: dict):
        for net_name, grads in update_info.items():
            for p, grad in zip(self.networks.net_dict[net_name].parameters(), grads):
                p.grad = grad
        self.__update(list(update_info.keys()))

    def __update(self, update_list):
        tau = self.tau
        for net_name in update_list:
            self.networks.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in update_list:
                for p, p_targ in zip(
                        self.networks.net_dict[net_name].parameters(),
                        self.networks.target_net_dict[net_name].parameters(),
                ):
                    p_targ.data.mul_(1 - tau)
                    p_targ.data.add_(tau * p.data)
    
    def __adapt_weights(self, iteration):
        assert self.weight_loss_v >= 0 and self.weight_loss_v <= 1, print("weight_loss_v =", self.weight_loss_v)
        if self.adapt_weights:
            self.w_loss_v = min(iteration/self.weights_saturate_iter, 1) * self.weight_loss_v
        else:
            self.w_loss_v = self.weight_loss_v

    def __compute_gradient(self, data, iteration):
        update_list = []

        start_time = time.time()

        self.__adapt_weights(iteration)
        self.networks.v.zero_grad()
        loss_critic, v, loss_v, loss_costate, loss_ratio = self.__compute_loss_v(data)
        loss_critic.backward()
        self.tb_info[tb_tags["loss_critic"]] = loss_critic.item()
        self.tb_info[tb_tags["critic_avg_value"]] = v.item()
        self.tb_info[tb_tags["loss_v"]] = loss_v.item()
        self.tb_info[tb_tags["loss_costate"]] = loss_costate.item()
        self.tb_info[tb_tags["loss_ratio"]] = loss_ratio
        update_list.append("v")
        self.networks.policy.zero_grad()
        loss_policy = self.__compute_loss_policy(data)
        loss_policy.backward()
        self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        update_list.append("policy")

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        return update_list

    def __compute_loss_v(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        o_init = deepcopy(o)
        v = self.networks.v(o_init)
        costate = v[:, 1:]
        v = v[:, 0]
        o_init.requires_grad = True
        info = data
        backup = 0
 
        o2 = o_init
        for step in range(self.forward_step):
            o = o2
            a = self.networks.policy(o)
            o2, r, d, info = self.envmodel.forward(o, a, d, info)
            backup +=  self.gamma ** step * (-r) * self.reward_scale
            
        v_target = backup + self.gamma ** self.forward_step * self.networks.v_target(o2)[:, 0]
        v_error = (v - v_target).unsqueeze(-1)
        loss_v = (v_error ** 2).mean()
        if self.equilibrium is not None:
            equilibrium = torch.tensor(self.equilibrium, dtype=torch.float32).reshape(1, -1)
            loss_v += (self.networks.v(equilibrium)[:, 0] ** 2).mean()

        lambda_n = self.networks.v_target(o2)[:, 1:].detach()
        fake_Hamiltonian = backup + self.gamma ** self.forward_step * torch.sum(o2 * lambda_n, dim=1)
        costate_target = torch.autograd.grad(fake_Hamiltonian.sum(), o_init, retain_graph=True)[0].detach()
        costate_error = (costate - costate_target).unsqueeze(-1)
        loss_costate = (costate_error ** 2).mean()
        if self.equilibrium is not None:
            equilibrium = torch.tensor(self.equilibrium, dtype=torch.float32).reshape(1, -1)
            equilibrium.requires_grad = True
            loss_costate += (self.networks.v(equilibrium)[:, 1:] ** 2).mean()
        loss_ratio = loss_v.item() / loss_costate.item()
        if loss_ratio >= 1:
            loss_v_scaled = loss_v / loss_ratio
            loss_costate_scaled = loss_costate
        else:
            loss_v_scaled = loss_v
            loss_costate_scaled = loss_costate * loss_ratio
        loss_critic = self.w_loss_v * loss_v_scaled + (1 - self.w_loss_v) * loss_costate_scaled

        return loss_critic, torch.mean(v), loss_v, loss_costate, loss_ratio

    def __compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        o_init = deepcopy(o)
        o_init.requires_grad = True
        o2 = o_init
        info = data
        fake_Hamiltonian = 0

        for p in self.networks.v.parameters():
            p.requires_grad = False

        for step in range(self.forward_step):
            o = o2
            a = self.networks.policy(o)
            o2, r, d, info = self.envmodel.forward(o, a, d, info)
            fake_Hamiltonian +=  self.gamma ** step * (-r) * self.reward_scale

        lambda_n = self.networks.v(o2)[:, 1:].detach()
        fake_Hamiltonian += self.gamma ** self.forward_step * torch.sum(o2 * lambda_n, dim=1)
        
        for p in self.networks.v.parameters():
            p.requires_grad = True

        return fake_Hamiltonian.mean()


if __name__ == "__main__":
    print("11111")
