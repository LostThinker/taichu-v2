from utils.utils import episode_unroller, action_choose, get_time_step, convert_to_tensor, null_padding
import torch
import time
from collector.base_collector import BaseCollector


class Collector(BaseCollector):
    def __init__(
            self,
            env,
            policy,
            collect_config,
            tb_logger=None
    ):
        self.env = env
        self.env_num = env.env_num
        self.policy = policy
        self.traj_buffer = {env_id: [] for env_id in range(self.env_num)}
        self.episode_buffer = []
        self.collect_config = collect_config
        self.tb_logger = tb_logger
        self.collect_count = 0
        self.total_envstep_count = 0
        self.total_episode_count = 0
        self.total_train_sample_count = 0  # 将episode按照unroll_len切片后的数量
        self.start_time = time.time()

    def collect(self, sample_num=None, unroll_len=None, collect_type=None):
        """

        Args:
            sample_num:
            unroll_len: 一个样本的time_step_length
            collect_type: 包括'unroll'和'episode'，'unroll'将一整个episode分割成unroll_len长度的片段，每一个片段为一个sample;
            'episode'则是一整个episode作为一个sample
        Returns:

        """
        if sample_num is None:
            sample_num = self.collect_config.n_sample
        if unroll_len is None:
            unroll_len = self.collect_config.unroll_len
        if collect_type is None:
            collect_type = self.collect_config.sample_type

        self.collect_count += 1

        return_data = []
        collected_num = 0
        start_time = time.time()
        start_total_envstep_count = self.total_envstep_count

        batch_obs = self.env.reset()
        self.policy.reset()
        while collected_num < sample_num:
            with torch.no_grad()
                policy_out = self.policy.inference(batch_obs)
            batch_prev_state = policy_out['prev_state']
            batch_specific = policy_out['specific']
            batch_action = action_choose

