import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Union, Mapping, List, NamedTuple, Tuple, Callable, Optional, Any


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


def dict_state_to_tensor(dict_state, device='cpu'):
    if dict_state is None:
        return dict_state
    else:
        rnn_type = dict_state[0]['rnn_type']
        if rnn_type == 'lstm':
            lstm_h = []
            lstm_c = []
            for env_id in range(len(dict_state.keys())):
                h, c = dict_state[env_id]['lstm_h'][None, ...], dict_state[env_id]['lstm_c'][None, ...]
                if isinstance(h, np.ndarray):
                    h = torch.from_numpy(h)
                    c = torch.from_numpy(c)
                lstm_h.append(h)
                lstm_c.append(c)
            lstm_h = torch.cat(lstm_h, dim=1).to(device)
            lstm_c = torch.cat(lstm_c, dim=1).to(device)
            state = (lstm_h, lstm_c)
            return state
        else:
            state = []
            for env_id in range(len(dict_state.keys())):
                hidden_state = dict_state[env_id]['hidden_state'][None, ...]
                if isinstance(hidden_state, np.ndarray):
                    hidden_state = torch.from_numpy(hidden_state)
                state.append(hidden_state)
            state = torch.cat(state, dim=1).to(device)
            return state


def batch_data_processor(batch, cat_1dim=True, device=torch.device("cpu")):
    """
    用于处理vec_env传来的n个智能体的batch观测数据，返回(env_num,agent_num,N)形状的数据，并保留原始的key
    Args:
        batch:
        cat_1dim:

    Returns:

    """
    assert batch is not None, batch
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        if elem.shape == (1,) and cat_1dim:
            return torch.cat(batch, 0).to(device)
        else:
            return torch.stack(batch, 0).to(device)
    elif isinstance(elem, np.ndarray):
        return batch_data_processor([torch.as_tensor(b) for b in batch], cat_1dim=cat_1dim, device=device)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32).to(device)
    elif isinstance(elem, int):
        return torch.tensor(batch, dtype=torch.int64).to(device)
    elif isinstance(elem, bool):
        return torch.tensor(batch, dtype=torch.bool).to(device)
    elif isinstance(elem, dict):
        ret = {}
        for key in elem:
            if key == 'rnn_type':
                ret[key] = batch[0][key]
            else:
                test_list = [d[key] for d in batch]
                ret[key] = batch_data_processor([d[key] for d in batch], cat_1dim=cat_1dim, device=device)
        return ret
    elif isinstance(elem, list):
        transposed = zip(*batch)
        return [batch_data_processor(samples, cat_1dim=cat_1dim, device=device) for samples in transposed]
    elif isinstance(elem, str):
        return elem
    elif elem is None:
        return elem
    raise TypeError('type not supported')


def action_choose(logit, action_mask=None)


def tensor_to_dict_state(data, rnn_type, batch_size, agent_num, device='cpu'):
    if rnn_type == 'lstm':
        h, c = data
        if device == 'cuda':
            h = h.to(device)
            c = c.to(device)
        else:
            h = h.cpu().numpy()
            c = c.cpu().numpy()
        state_dict = {}
        for env_id in range(batch_size):
            state_dict[env_id] = {"rnn_type": rnn_type,
                                  "lstm_h": h[:, env_id * agent_num:(env_id + 1) * agent_num, :][0],
                                  "lstm_c": c[:, env_id * agent_num:(env_id + 1) * agent_num, :][0]}


def prev_state_split(data):
    if data['rnn_type'] == 'lstm':
        prev_state = []
        for i in range(data['lstm_h'].shape[0]):
            prev_state.append((data['lstm_h'][i].unsqueeze(0), data['lstm_c'][i].unsqueeze(0)))
    else:
        prev_state = []
        for i in range(data['hidden_state'].shape[0]):
            prev_state.append(data['hidden_state'][i].unsqueeze(0))
    return prev_state


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


def one_hot(val: torch.LongTensor, num: int, num_first: bool = False) -> torch.FloatTensor:
    r"""
    Overview:
        Convert a ``torch.LongTensor`` to one hot encoding.
        This implementation can be slightly faster than ``torch.nn.functional.one_hot``
    Arguments:
        - val (:obj:`torch.LongTensor`): each element contains the state to be encoded, the range should be [0, num-1]
        - num (:obj:`int`): number of states of the one hot encoding
        - num_first (:obj:`bool`): If ``num_first`` is False, the one hot encoding is added as the last; \
            Otherwise as the first dimension.
    Returns:
        - one_hot (:obj:`torch.FloatTensor`)
    Example:
        # >>> one_hot(2*torch.ones([2,2]).long(),3)
        # tensor([[[0., 0., 1.],
        #          [0., 0., 1.]],
        #         [[0., 0., 1.],
        #          [0., 0., 1.]]])
        # >>> one_hot(2*torch.ones([2,2]).long(),3,num_first=True)
        # tensor([[[0., 0.], [1., 0.]],
        #         [[0., 1.], [0., 0.]],
        #         [[1., 0.], [0., 1.]]])
    """
    assert (isinstance(val, torch.Tensor)), type(val)
    assert val.dtype == torch.long
    assert (len(val.shape) >= 1)
    old_shape = val.shape
    val_reshape = val.reshape(-1, 1)
    ret = torch.zeros(val_reshape.shape[0], num, device=val.device)
    # To remember the location where the original value is -1 in val.
    # If the value is -1, then it should be converted to all zeros encodings and
    # the corresponding entry in index_neg_one is 1, which is used to transform
    # the ret after the operation of ret.scatter_(1, val_reshape, 1) to their correct encodings bellowing
    index_neg_one = torch.eq(val_reshape, -1).long()
    if index_neg_one.sum != 0:  # 如果val中有-1
        # 将-1变成0
        val_reshape = torch.where(
            val_reshape != -1, val_reshape,
            torch.zeros(val_reshape.shape, device=val.device).long()
        )
    try:
        ret.scatter_(1, val_reshape, 1)
        if index_neg_one.sum() != 0:  # 如果val中有-1
            ret = ret * (1 - index_neg_one)  # 把-1的编码从[1,0,...,0] to [0,0,...,0]
    except RuntimeError:
        raise RuntimeError('value: {}\nnum: {}\t:val_shape: {}\n'.format(val_reshape, num, val_reshape.shape))
    if num_first:
        return ret.permute(1, 0).reshape(num, *old_shape)
    else:
        return ret.reshape(*old_shape, num)


def parallel_wrapper(forward_fn: Callable) -> Callable:
    r"""
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch.
    Arguments:
        - forward_fn (:obj:`Callable`): Normal ``nn.Module`` 's forward function.
    Returns:
        - wrapper (:obj:`Callable`): Wrapped function.
    """

    def wrapper(x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        T, B = x.shape[:2]

        def reshape(d):
            if isinstance(d, list):
                d = [reshape(t) for t in d]
            elif isinstance(d, dict):
                d = {k: reshape(v) for k, v in d.items()}
            else:
                d = d.reshape(T, B, *d.shape[1:])
            return d

        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
        return x
    return wrapper


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

def get_activation_fn(activation):
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'tanh':
        return nn.Tanh()


class objDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = objDict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def get_optimizer(optim):
    if isinstance(optim, str):
        if optim == 'adam':
            optim = torch.optim.Adam
        elif optim == 'sgd':
            optim = torch.optim.SGD
        elif optim == 'rmsprop':
            optim = torch.optim.RMSprop
        else:
            raise NotImplementedError
    else:
        pass
    return optim


def target_net_update(target_net, net, theta=0.001):
    net_state_dict = net.state_dict()
    for name, p in target_net.named_parameters():
        p.data = (1 - theta) * p.data + theta * net_state_dict[name]
    return target_net


def get_env_config(env_config=None):
    """

    Args:
        env_args: 包含env_type,以及相关的环境初始化参数,不同的env_type参数格式也不同

    Returns: {env_type,env_args,env_fn,env_info}

    """
    if env_config is not None:
        if env_config["env_type"] == 'SC2':
            from envs.env_wrapper import GroupSC2Env
            env = GroupSC2Env(**env_config["env_args"])
            env_info = env.get_env_info()
            env_config['env_info'] = env_info
            env_config['env_fn'] = GroupSC2Env
            return env_config
        elif env_config["env_type"] == 'particle':
            pass
    else:
        default_env_config = dict(
            env_type='SC2',
            env_args=dict(  # 用于直接传入环境类中进行初始化的参数，需保持接口一致
                map_name='8m',
                difficulty=7,
                seed=0,
                max_step=500
            )
        )
        from envs.env_wrapper import GroupSC2Env
        env = GroupSC2Env(**default_env_config["env_args"])
        env_info = env.get_env_info()
        default_env_config['env_info'] = env_info
        return default_env_config
