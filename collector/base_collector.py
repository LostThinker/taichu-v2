# -*- coding:utf-8 -*-
import sys

sys.path.append("..")
import numpy as np
from abc import ABC, abstractmethod
from envs.env_wrapper import GroupMaAtariEnv
from envs.vec_env_wrapper import VecEnvWrapper
from utils.utils import make_env, make_policy, make_buffer, vec_env_data_process


class BaseCollector(ABC):
    def __init__(
            self,
            env_args,
            policy_args,
            buffer_args
    ):
        self.env_args = env_args
        self.policy_args = policy_args
        self.buffer_args = buffer_args
        self.env = make_env(env_args)
        self.policy = make_policy(policy_args)
        self.buffer = make_buffer(buffer_args)

    @abstractmethod
    def collect(self, commander, batch_size, collect_type='step'):
        """
        根据传来的模型与环境交互采样，可根据step数量或者episode数量采样，并将采集的样本放入buffer中

        collect_type：'step' or 'episode'

        """
        if collect_type == 'step':
            pass
        elif collect_type == 'episode':
            pass
        pass

    def get_batch(self, batch_size):
        """
        从buffer中采样数据，如果是on-policy可以返回整个buffer的数据
        """

        return self.buffer.get(batch_size)


def vec_env_process_test():
    env_num = 4
    env_fn = GroupMaAtariEnv
    vec_env = VecEnvWrapper(env_num, env_fn)
    batch_obs, batch_camp_shared_info = vec_env.reset()
    vec_action_dcit = {}
    for env_id in range(env_num):
        action_dict = {}
        for camp_id, group in vec_env.camp.items():
            action_dict[camp_id] = {}
            for group_id, units_id_list in group.items():
                action_dict[camp_id][group_id] = {}
                for unit_id in units_id_list:
                    action_dict[camp_id][group_id][unit_id] = vec_env.units_info['unit_action_space_list'][
                        unit_id].sample()
        vec_action_dcit[env_id] = action_dict

    batch_obs, batch_reward, batch_done, batch_truncate, batch_info, batch_camp_shared_info, batch_game_over = vec_env.step(
        vec_action_dcit)
    out = vec_env_data_process(batch_obs)

    print('a')


if __name__ == '__main__':
    vec_env_process_test()
