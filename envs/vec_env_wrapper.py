from collections import namedtuple
import numpy as np
from multiprocessing import Process, Manager, Pipe
from functools import partial
from envs.env_wrapper import GroupMaAtariEnv


# Step_return = namedtuple('Step_return', ['obs', 'reward', 'done', 'info'])
# Reset_return = namedtuple('Reset_return', ['obs'])
# import torch


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnvWrapper:
    def __init__(self, env_num, env_fn, env_args={}):
        self.env_num = env_num
        self.env_fn = env_fn  # 非实例化
        self.env_args = env_args
        self.parent_pipes = {}
        self.child_pipes = {}
        self.processes = {}
        self.total_step = 0
        self.total_episode = 0
        self.done = {env_id: False for env_id in range(self.env_num)}
        self._get_env_info()
        for env_id in range(env_num):
            self.create_process(env_id)

    def create_process(self, env_id):
        self.parent_pipes[env_id], self.child_pipes[env_id] = Pipe()
        self.processes[env_id] = Process(target=worker_fn, args=(
            self.parent_pipes[env_id],
            self.child_pipes[env_id],
            CloudpickleWrapper(partial(self.env_fn, **self.env_args))
        ),
                                         daemon=True
                                         )
        self.processes[env_id].start()
        self.child_pipes[env_id].close()

    def reset(self):
        batch_obs = {env_id: None for env_id in range(self.env_num)}
        batch_camp_shared_info = {env_id: None for env_id in range(self.env_num)}
        try:
            for env_id in range(self.env_num):
                self.parent_pipes[env_id].send(['reset', [], {}])

            for env_id in range(self.env_num):
                obs_dict, camp_shared_info_dict = self.parent_pipes[env_id].recv()['reset']
                batch_obs[env_id] = obs_dict
                batch_camp_shared_info[env_id] = camp_shared_info_dict

            # 处理数据
            batch_obs = self._process_batch_data(batch_obs)

            batch_camp_shared_info = self._process_batch_camp_shared_info(batch_camp_shared_info)

            return batch_obs, batch_camp_shared_info
        except BaseException as e:
            raise e

    def step(self, actions):
        batch_obs = {env_id: None for env_id in range(self.env_num)}
        batch_reward = {env_id: None for env_id in range(self.env_num)}
        batch_done = {env_id: None for env_id in range(self.env_num)}
        batch_truncate = {env_id: None for env_id in range(self.env_num)}
        batch_info = {env_id: None for env_id in range(self.env_num)}
        batch_camp_shared_info = {env_id: None for env_id in range(self.env_num)}
        batch_game_over = {env_id: None for env_id in range(self.env_num)}

        try:
            for env_id in range(self.env_num):
                self.parent_pipes[env_id].send(['step', [actions[env_id]], {}])

            for env_id in range(self.env_num):
                ret = self.parent_pipes[env_id].recv()
                next_obs_dict, reward_dict, done_dict, truncate_dict, info_dict, camp_shared_info_dict, game_over = ret[
                    "step"]

                batch_obs[env_id] = next_obs_dict
                batch_reward[env_id] = reward_dict
                batch_done[env_id] = done_dict
                batch_truncate[env_id] = truncate_dict
                batch_info[env_id] = info_dict
                batch_camp_shared_info[env_id] = camp_shared_info_dict
                batch_game_over[env_id] = game_over

                # logging
                self.total_step += 1
                if game_over:
                    self.total_episode += 1

            batch_obs = self._process_batch_data(batch_obs)
            batch_reward = self._process_batch_data(batch_reward)
            batch_done = self._process_batch_data(batch_done)
            batch_truncate = self._process_batch_data(batch_truncate)
            batch_info = self._process_batch_data(batch_info)
            batch_camp_shared_info = self._process_batch_camp_shared_info(batch_camp_shared_info)

            return batch_obs, batch_reward, batch_done, batch_truncate, batch_info, batch_camp_shared_info, batch_game_over

        except BaseException as e:
            self.close()
            raise e

    def close(self):
        try:
            for env_id in range(self.env_num):
                self.parent_pipes[env_id].send(['close', [], {}])

            for env_id in range(self.env_num):
                self.processes[env_id].join()

        except BaseException as e:
            raise e

    def _get_env_info(self):
        env = self.env_fn(**self.env_args)
        self.units_info = env.units_info
        self.camp = env.camp
        self.camp_num = env.camp_num
        self.group_num = env.group_num
        self.group_type = env.group_type
        self.game_name = env.game_name
        env.close()

    def get_total_steps(self):
        data = {
            "total_step": self.total_step,
            "total_episode": self.total_episode
        }
        return data

    def _process_batch_data(self, batch_data):
        batch_vec_data = batch_data[0]
        for camp_id, group in batch_vec_data.items():
            for group_id, units in group.items():
                # 添加env_id
                batch_vec_data[camp_id][group_id] = {0: units}

        for env_id in range(1, self.env_num):
            env_data = batch_data[env_id]
            for camp_id, group in env_data.items():
                for group_id, units in group.items():
                    batch_vec_data[camp_id][group_id][env_id] = units
        return batch_vec_data

    def _process_batch_camp_shared_info(self, batch_camp_shared_info):
        batch_vec_data = batch_camp_shared_info[0]
        for camp_id, camp_info in batch_vec_data.items():
            batch_vec_data[camp_id] = {0: camp_info}

        for env_id in range(1, self.env_num):
            for camp_id, camp_info in batch_camp_shared_info[env_id].items():
                batch_vec_data[camp_id][env_id] = camp_info

        return batch_vec_data


def worker_fn(parent_pipe, child_pipe, env_fn_wrapper):
    # torch.set_num_threads(1)
    env_fn = env_fn_wrapper.x
    env = env_fn()
    parent_pipe.close()

    def step_fn(action):
        try:
            # next_obs_dict, reward_dict, done_dict,truncate_dict, info_dict, camp_shared_info_dict =
            ret = env.step(action)
            ret = {"step": ret}
            return ret
        except BaseException as e:
            env.close()
            raise e

    def reset_fn(*args, **kwargs):
        try:
            ret = env.reset(*args, **kwargs)
            ret = {"reset": ret}
            return ret
        except BaseException as e:
            env.close()
            raise e

    while True:
        try:
            cmd, args, kwargs = child_pipe.recv()
        except EOFError:  # for the case when the pipe has been closed
            child_pipe.close()
            break
        try:
            if cmd == 'close':
                env.close()
                break
            else:
                if cmd == 'step':
                    ret = step_fn(*args)
                elif cmd == 'reset':
                    ret = reset_fn(*args, **kwargs)
                    # last_obs = ret['reset'].obs
                else:
                    print("errors")
                    ret = None
                child_pipe.send(ret)
        except EOFError:  # for the case when the pipe has been close
            child_pipe.close()
            break

    print('env closed')


def vec_env_test():
    env_num = 4
    env_fn = GroupMaAtariEnv
    vec_env = VecEnvWrapper(env_num, env_fn)
    returns = vec_env.reset()
    print("a")


if __name__ == '__main__':
    vec_env_test()
