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
            with torch.no_grad():
                policy_out = self.policy.inference(batch_obs)
            batch_prev_state = policy_out['prev_state']
            batch_specific = policy_out['specific']
            batch_action = action_choose(policy_out['logit'], policy_out['action_mask'], epsilon=self._get_epsilon(),
                                         choose_type='epsilon').squeeze(0).numpy()
            batch_next_obs, batch_reward, batch_done, batch_info = self.env.step(batch_action)
            for env_id in range(self.env_num):
                if batch_reward[env_id] is None:
                    pass
                else:
                    if batch_info[env_id]['env_state'] == 'abnormal':
                        self.total_envstep_count -= len(self.traj_buffer[env_id])
                        self.traj_buffer[env_id] = []
                        batch_done[env_id] = False  # 环境full_restart
                        self.policy.reset_state(env_id)
                    else:
                        time_step = get_time_step(
                            batch_obs[env_id], batch_action[env_id], batch_next_obs[env_id],
                            batch_reward[env_id], batch_done[env_id], batch_prev_state[env_id],
                            batch_specific[env_id]
                        )
                        self.total_envstep_count += 1
                        self.traj_buffer[env_id].append(time_step)
                if batch_done[env_id] is True:
                    self.policy.reset_state(env_id)
                    self.total_episode_count += 1
                    episode = self.traj_buffer[env_id]

                    if collect_type == 'unroll':
                        unroll_data = episode_unroller(episode, unroll_len=unroll_len)
                        if not unroll_data:
                            pass
                        else:
                            self.total_train_sample_count += len(unroll_len)
                            collected_num += len(unroll_data)
                            return_data.extend(unroll_data)
                    elif collect_type == 'episode':
                        return_data.append(episode)
                        collected_num += 1
                    self.traj_buffer[env_id] = []

            batch_obs = batch_next_obs

        if collect_type == 'episode':
            return_data = null_padding(return_data)  # episode长度相同
        elif collect_type == 'unroll':
            return_data = return_data[:sample_num]

        end_time = time.time()
        end_total_envstep_count = self.total_envstep_count
        time_used = (end_time - start_time)
        sampled_env_step = end_total_envstep_count - start_total_envstep_count

        sample_speed = sampled_env_step / time_used

        collect_info = dict(
            info_type='collect_info',
            collect_count=self.collect_count,
            total_envstep_count=self.total_envstep_count,
            collect_info=dict(
                sample_speed=sample_speed,
                time_used=time_used,
                sampled_env_step=sampled_env_step
            )
        )
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('collector/sample_speed(step/s)', sample_speed, self.collect_count)
            self.tb_logger.add_scalar('collector/time_used', time_used, self.collect_count)
            self.tb_logger.add_scalar('collector/sampled_env_step', sampled_env_step, self.collect_count)

        return return_data, collect_info

    def close(self):
        self.env.close()

    def _get_epsilon(self):
        if self.total_envstep_count >= self.collect_config.epsilon_step:
            epsilon = self.collect_config.epsilon_end
        else:
            epsilon = self.collect_config.epsilon_start - (
                    self.collect_config.epsilon_start - self.collect_config.epsilon_end) * self.total_envstep_count / self.collect_config.epsilon_step
        return epsilon

    def update_policy(self, state_dict):
        self.policy.update_model_params(state_dict)
