import numpy as np
import cv2


def make_env(env_arg):
    """
    根据参数返回环境实例
    """
    pass


def make_policy(policy_arg):
    """
    根据参数返回策略实例
    """
    pass


def make_buffer(buffer_arg):
    """
    根据参数返回buffer实例
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


def rgb_to_gray(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    return obs


def resize(obs, shape):
    obs = cv2.resize(obs, shape)
    return obs


def gray_resize(obs, shape):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, shape)
    return obs
