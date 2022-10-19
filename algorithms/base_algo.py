import torch
from abc import ABC, abstractmethod
from utils.utils import make_policy


class BaseAlgo(ABC):
    def __init__(
            self,
            env_args,
            policy_args,
    ):
        self.env_args = env_args
        self.policy_args = policy_args
        self.policy = make_policy(env_args, policy_args)

    @abstractmethod
    def inference(self, data):
        """
        前向推理
        """
        pass

    @abstractmethod
    def train(self, data):
        """
        训练流程
        """
        pass

    @abstractmethod
    def _calculate_loss(self, data):
        """
        不同算法的loss计算，部分共用
        """
        pass
