from typing import Union, Optional, Dict, Callable, List, Tuple
from model.encoder import FCEncoder, ConvEncoder
from model.head import DiscreteHead
from utils.utils import one_hot, parallel_wrapper
import torch
import torch.nn as nn


def get_rnn(rnn_type, input_size, hidden_size, rnn_num_layers):
    if rnn_type == 'normal':
        model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_num_layers)
    elif rnn_type == 'lstm':
        model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_num_layers)
    elif rnn_type == 'gru':
        model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_num_layers)
    else:
        raise ValueError("rnn_type 只包含 'normal','lstm','gru'")
    return model


class DRQN(nn.Module):
    def __init__(
            self,
            obs_shape: Union[int, list],
            action_shape: int,
            hidden_size_list: Union[int, list],
            hidden_channel_list: list = [16, 32, 32],
            input_shape: Optional[int] = None,
            agent_num: Optional[int] = None,
            conv_output_shape: Union[int] = 128,
            kernel_size_list: list = [8, 4, 3],
            stride_list: list = [4, 2, 1],
            activation: Optional[nn.Module] = nn.ReLU(),
            rnn_type: Union[str] = 'normal',
            rnn_num_layers: Optional[int] = 1,
            rnn_hidden_size: Optional[int] = 128,
            head_layer_num: int = 1,
            norm_type: Optional[str] = None,
            noise: Optional[bool] = False,
    ):
        """
        Overview:
            DQN+RNN 用于包含RNN的actor网络
        Args:
            obs_shape: obs可以为一维向量和图像输入
            action_shape:单个智能体的动作维度
            # action_input_shape:此为其输入的action的维度，如果网络需要输入所有智能体的动作，该参数值应为单个动作维度乘上智能体数量,如果为None则不输入动作
            hidden_size_list:隐藏层的维度
            agent_num:如果为None，则不输入agent_id
            rnn_type:可选'normal','gru','lstm'
            activation:默认Relu
            norm_type:标准化方法，包括'batch','layer'
        """
        super(DRQN, self).__init__()
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.agent_num = agent_num
        self.encoder_input_shape = input_shape

        if isinstance(obs_shape, int):
            self.obs_type = 'vec'  # 向量输入
            self.encoder = FCEncoder(self.encoder_input_shape, hidden_size_list, activation=activation,
                                     norm_type=norm_type)
            self.encoder_feature_size = hidden_size_list[-1]
        elif isinstance(obs_shape, list):
            self.obs_type = 'img'  # 图像输入
            self.feature_encoder = ConvEncoder(obs_shape, conv_output_shape, hidden_channel_list, kernel_size_list,
                                               stride_list, activation)
            self.fc_encoder = FCEncoder(self.encoder_input_shape, hidden_size_list, activation=activation,
                                        norm_type=norm_type)
            self.encoder = nn.Sequential(self.feature_encoder, self.fc_encoder)
            self.encoder_feature_size = hidden_size_list[-1]
        else:
            raise ValueError('obs_shape 应为int或list')
        self.rnn = get_rnn(rnn_type, self.encoder_feature_size, rnn_hidden_size, rnn_num_layers)

        self.head = DiscreteHead(rnn_hidden_size, action_shape, head_layer_num, activation, norm_type, noise)

    def forward(
            self,
            obs: torch.Tensor,
            hidden_state: Union[torch.Tensor, Tuple] = None,
            last_action: Optional[torch.Tensor] = None,
            agent_id: Optional[torch.Tensor] = None,
            inference=False,
    ):
        """
        Overview:

        Args:
            inputs: (time_step,batch_size*n_agent,obs_dim),输入为连续的time_step步的观测
            hidden_state:
            inference:

        Returns:

        """
        self.rnn.flatten_parameters()  # 减小内存消耗
        obs, hidden_state, last_action, agent_id = self._data_process(obs, hidden_state,
                                                                      last_action, agent_id)  # to T, B*A, N
        if self.rnn_type == 'normal' or self.rnn_type == 'gru':
            assert isinstance(hidden_state, torch.Tensor)
        elif self.rnn_type == 'lstm':
            assert isinstance(hidden_state, Tuple)

        x = self.get_rnn_inputs(obs, last_action, agent_id)
        prev_state = hidden_state
        rnn_out_list = []
        hidden_state_list = []

        if self.rnn_type == 'normal' or self.rnn_type == 'gru':
            for t in range(x.shape[0]):
                outputs, hidden_state = self.rnn(x[t:t + 1], hidden_state)
                hidden_state_list.append(hidden_state)
                rnn_out_list.append(outputs)

            hidden_state_out = torch.cat(hidden_state_list, dim=-3)
            next_state_out = hidden_state

        elif self.rnn_type == 'lstm':
            lstm_h_list = []
            lstm_c_list = []
            for t in range(x.shape[0]):
                # if inference:
                outputs, hidden_state = self.rnn(x[t:t + 1], hidden_state)
                rnn_out_list.append(outputs)
                lstm_h_list.append(hidden_state[0])
                lstm_c_list.append(hidden_state[1])
            lstm_h = torch.cat(lstm_h_list, dim=-3)
            lstm_c = torch.cat(lstm_c_list, dim=-3)
            hidden_state_out = {"lstm_h": lstm_h, "lstm_c": lstm_c}
            next_state_out = hidden_state
        x = torch.cat(rnn_out_list, 0)  # 在timestep维度上堆叠，形成(time_step, batch_size*n_agent, obs_dim)的原始形状
        x = parallel_wrapper(self.head)(x)
        x['prev_state'] = prev_state
        x['next_state'] = next_state_out
        x['hidden_state'] = hidden_state_out
        x['rnn_type'] = self.rnn_type
        return x

    def get_rnn_inputs(self, obs, last_action, agent_id, inference=False):
        if self.obs_type == 'vec':
            inputs = obs
            if last_action is not None:
                inputs = torch.cat([inputs, last_action], -1)
            if agent_id is not None:
                if len(agent_id.shape) != len(obs.shape):
                    assert self.agent_num is not None
                    agent_id = one_hot(agent_id, self.agent_num)
                inputs = torch.cat([inputs, agent_id], -1)
        elif self.obs_type == 'img':
            if inference:
                obs_feature = self.feature_encoder(obs)
            else:
                obs_feature = parallel_wrapper(self.feature_encoder)(obs)
            inputs = obs_feature
            if last_action is not None:
                inputs = torch.cat([inputs, last_action], -1)
            if agent_id is not None:
                if len(agent_id.shape) != len(obs.shape) - 2:
                    assert self.agent_num is not None
                    agent_id = one_hot(agent_id, self.agent_num)
                inputs = torch.cat([inputs, agent_id], -1)
        else:
            raise ValueError("")
        if inference:
            inputs = self.encoder(inputs)
        else:
            inputs = parallel_wrapper(self.encoder)(inputs)
            inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1])
        return inputs

    def _data_preprocess(self, obs, hidden_state, last_action, agent_id):
        device = obs.device.type
        if len(obs.shape) == 6:
            obs = obs.reshape(obs.shape[0], -1, *obs.shape[3:])
        if len(obs.shape) == 4:
            obs = obs.reshape(obs.shape[0], -1, obs.shape[-1])

        if hidden_state is None:
            if self.rnn_type == 'lstm':
                h = torch.zeros(1, obs.shape[1], self.rnn_hidden_size).to(device)
                c = torch.zeros(1, obs.shape[1], self.rnn_hidden_size).to(device)
                hidden_state = (h, c)
            else:
                hidden_state = torch.zeros(1, obs.shape[1], self.rnn_hidden_size).to(device)

        elif isinstance(hidden_state, Tuple):
            if len(hidden_state[0].shape) == 4:
                hidden_state = (hidden_state[0].reshape(hidden_state[0].shape[0], -1, hidden_state[0].shape[-1]),
                                hidden_state[1].reshape(hidden_state[1].shape[0], -1, hidden_state[1].shape[-1]))

        elif isinstance(hidden_state, torch.Tensor):
            if len(hidden_state.shape) == 4:
                hidden_state = hidden_state.reshape(hidden_state.shape[0], -1, last_action.shape[-1])

        if len(last_action.shape) == 4:
            last_action = last_action.reshape(last_action.shape[0], -1, last_action.shape[-1])
        if len(agent_id.shape) == 4:
            agent_id = agent_id.reshape(agent_id.shape[0], -1, agent_id.shape[-1])
        return obs, hidden_state, last_action, agent_id
