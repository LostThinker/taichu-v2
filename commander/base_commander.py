import torch
from abc import ABC, abstractmethod


class GeneralBaseCommander(ABC):
    def __init__(
            self,
            env_args,
            group_policy_args,

    ):
        self.env_args = env_args
        self.group_policy_args = group_policy_args
        self.policy_pool = []  # 考虑加入种群训练
        self._generate_policy()

    def _generate_policy(self):
        """
        根据group_policy_args生成policy实例并放入policy_pool中，idx为group_id
        """
        pass

    def command(self, obs_dict):
        """
        与环境交互推理，返回动作
        """
        command_data = {}
        for camp_id, camp_data in obs_dict.items():
            command_data[camp_id] = {}
            for group_id, group_data in camp_data.items():
                batch_obs = group_data['obs']  # (env_id,agent_num,N)
                batch_interact_data_dict = self.policy_pool[group_id].inference(batch_obs)
                # batch_interact_data_dict 包含一系列需要存储的交互数据
                command_data[camp_id][group_id] = batch_interact_data_dict
        return command_data

    @abstractmethod
    def train(self, data):
        """
        可实现种群训练
        """
        for policy in self.policy_pool:
            policy.train(data)
