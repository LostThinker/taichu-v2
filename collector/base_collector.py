import numpy as np
from abc import ABC, abstractmethod
from envs.env_wrapper import GroupMaAtariEnv
from envs.vec_env_wrapper import VecEnvWrapper

class BaseCollector(ABC):
    def __init__(
            self,
            policy,
            env_args
    ):
        self.env_args = env_args
        self.env = self._make_env(env_args)

    def _make_env(self, env_args):
        """

        """
        pass

    @abstractmethod
    def collect(self,state_dict=None):
        """

        Returns:

        """
        pass




def vec_env_data_process(data_dict):
    for camp_id, camp in data_dict.items():
        for group_id, group in camp.items():
            vec_group_data = {}
            for env_id, unit_dict in group.items():
                group_data = {}
                for unit_id, unit_data in unit_dict.items():
                    for unit_data_key, unit_data_value in unit_data.items():
                        group_data_value = group_data.get(unit_data_key)
                        if group_data_value is None:
                            group_data[unit_data_key] = [unit_data_value]
                        else:
                            group_data_value.append(unit_data_value)
                for data_key, data_value in group_data.items():
                    if vec_group_data.get(data_key) is None:
                        vec_group_data[data_key] = [data_value]
                    else:
                        vec_group_data[data_key].append(data_value)
            for data_key, data_value in vec_group_data.items():
                vec_group_data[data_key] = np.array(data_value)
            data_dict[camp_id][group_id] = vec_group_data
    return data_dict


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
    # for camp_id, camp in batch_obs.items():
    #     for group_id, group in camp.items():
    #         vec_group_obs = {}
    #         for env_id, unit_dict in group.items():
    #             group_obs = {}
    #             for unit_id, unit_data in unit_dict.items():
    #                 for unit_data_key, unit_data_value in unit_data.items():
    #                     group_data_value = group_obs.get(unit_data_key)
    #                     if group_data_value is None:
    #                         group_obs[unit_data_key] = [unit_data_value]
    #                     else:
    #                         group_data_value.append(unit_data_value)
    #             for data_key, data_value in group_obs.items():
    #                 vec_group_data = vec_group_obs.get(data_key)
    #                 if vec_group_data is None:
    #                     vec_group_obs[data_key] = [data_value]
    #                 else:
    #                     vec_group_obs[data_key].append(data_value)
    #         for data_key, data_value in vec_group_obs.items():
    #             vec_group_obs[data_key] = np.array(data_value)
    #         batch_obs[camp_id][group_id] = vec_group_obs

    print('a')


# def dict_to_list(dict_data):


if __name__ == '__main__':
    vec_env_process_test()
