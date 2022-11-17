from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
from model.block import ResFCBlock, fc_block, ResBlock


class FCEncoder(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_size_list: Union[int, list],
                 activation: Optional[nn.Module] = nn.ReLU(),
                 res_block: bool = False,
                 norm_type: Optional[str] = None
                 ):
        """
        Overview:
            用于对向量输入通过全连接网络提取特征
        Args:
            input_shape:
            hidden_size_list:
            res_block: 可选添加Residual Block
            activation:
            norm_type: 标准化方法，包括batch norm:'BN',layer norm:'LN'
        """
        super(FCEncoder, self).__init__()
        self.input_shape = input_shape
        self.act = activation
        self.model_list = []
        self.model_list.append(nn.Linear(input_shape, hidden_size_list[0]))
        if activation is not None:
            self.model_list.append(activation)
        if isinstance(hidden_size_list, int):
            hidden_size_list = [hidden_size_list]
        if res_block:
            assert len(set(hidden_size_list)) == 1, 'Res_block只支持输入输出维度相同，因此hidden_size_list里的值需要相同'
            for i in range(len(hidden_size_list)):
                self.model_list.append(
                    ResFCBlock(in_channels=hidden_size_list[i], activation=activation,
                               norm_type=norm_type)
                )
        else:
            for i in range(len(hidden_size_list) - 1):
                self.model_list.append(fc_block(hidden_size_list[i], hidden_size_list[i + 1], activation, norm_type))
        self.model = nn.Sequential(*self.model_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(
            self,
            input_shape: Union[list, Tuple],
            output_shape: int,
            hidden_channel_list: list,
            kernel_size_list: list,
            stride_list: list,
            activation: Optional[nn.Module] = nn.ReLU(),
            res_block: bool = False,
            norm_type: Optional[str] = None,

    ):
        """
        Overview:
            使用卷积网络提取图像输入的特征
        Args:
            input_shape: [C,H,W]
            hidden_channel_list:各卷积层的output channel值
            kernel_size_list:
            stride_list:
            activation:
            res_block:如果选择则会在卷积的最后添加一个标准的Residual block
            norm_type:目前支持Batch norm,"BN"
        """
        super(ConvEncoder, self).__init__()
        self.input_shape = input_shape  # C,H,W
        self.output_shape = output_shape
        self.act = activation
        self.hidden_channel_list = hidden_channel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.model_list = []
        input_channel = input_shape[0]
        assert len({len(hidden_channel_list), len(kernel_size_list), len(stride_list)}) == 1
        for i in range(len(kernel_size_list)):
            self.model_list.append(nn.Conv2d(input_channel, hidden_channel_list[i],
                                             kernel_size_list[i], stride_list[i]))
            self.model_list.append(self.act)
            input_channel = hidden_channel_list[i]
        if res_block:
            self.model_list.append(ResBlock(input_channel, norm_type=norm_type))

        self.model_list.append(nn.Flatten())
        self.conv_model = nn.Sequential(*self.model_list)
        flatten_size = self._get_flatten_size()
        self.out_fc = nn.Linear(flatten_size, output_shape)

    def _get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self.input_shape)
        with torch.no_grad():
            output = self.conv_model(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_model(x)
        x = self.out_fc(x)
        return x
