from typing import Union, Optional, Dict, Callable, List, Tuple
from model.encoder import FCEncoder, ConvEncoder
from model.head import DiscreteHead
from utils.utils import
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


