from typing import Union, Optional, Tuple
import torch
import torch.nn as nn

############## FC Block ###############


def fc_block(
        in_channels: int,
        out_channels: int,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5
) -> nn.Sequential:
    """
    Overview:
        x -> fc -> norm -> act -> dropout -> out
    Args:
        in_channels:
        out_channels:
        activation:
        norm_type: 'BN' or 'LN'
        use_dropout:
        dropout_probability:

    Returns:
        nn.Sequential
    """
    model_list = [nn.Linear(in_channels, out_channels)]
    if norm_type is not None:
        if norm_type == 'BN':
            model_list.append(nn.BatchNorm1d(out_channels))
        elif norm_type == 'LN':
            model_list.append(nn.LayerNorm(out_channels))
        else:
            raise ValueError("norm_type 只包含'BN'和'LN'")
    if activation is not None:
        model_list.append(activation)
    if use_dropout:
        model_list.append(nn.Dropout(p=dropout_probability))
    model = nn.Sequential(*model_list)
    return model


def conv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple],
        activation: Optional[nn.Module] = nn.ReLU(),
        pooling_type: Optional[str] = None,
        pooling_kernel_size: Union[int, Tuple[int, int]] = 2,
        pooling_stride: Union[int, Tuple[int, int]] = None,
        pad_type: Optional[str] = None,
        pad_size: Union[int, Tuple] = 0,
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5
) -> nn.Sequential:
    """
    Overview:
        x -> Conv2d -> norm -> act -> dropout -> out
    Args:
        in_channels:
        out_channels:
        kernel_size:
        stride:
        activation:
        pooling_type:
        pooling_kernel_size:
        pooling_stride:
        pad_type:
        pad_size:
        norm_type:
        use_dropout:
        dropout_probability:

    Returns:
        nn.Sequential
    """
    model_list = []
    if pad_type is not None:
        if pad_type == 'zero':
            model_list.append(nn.ZeroPad2d(pad_size))
        elif pad_type == 'reflect':
            model_list.append(nn.ReflectionPad2d(pad_size))
        elif pad_type == 'replication':
            model_list.append(nn.ReplicationPad2d(pad_size))

    model_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    if pooling_type is not None:
        if pooling_type == 'max':
            model_list.append(nn.MaxPool2d(pooling_kernel_size, pooling_stride))
        elif pooling_stride == 'avg':
            model_list.append(nn.AvgPool2d(pooling_kernel_size, pooling_stride))
    if norm_type is not None:
        if norm_type == 'BN':
            model_list.append(nn.BatchNorm2d(out_channels))
        else:
            raise ValueError('目前只支持BatchNorm2d')
    if activation is not None:
        model_list.append(activation)
    if use_dropout:
        model_list.append(nn.Dropout(p=dropout_probability))

    model = nn.Sequential(*model_list)
    return model


class ResFCBlock(nn.Module):
    """
     Overview:
        Residual Block with 2 fully connected block
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+
    """

    def __init__(
            self,
            in_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm_type: str = 'BN'
    ):
        super(ResFCBlock, self).__init__()
        self.activation = activation
        self.model = nn.Sequential(
            fc_block(in_channels, in_channels, activation, norm_type),
            fc_block(in_channels, in_channels, None, norm_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.model(x)
        x = self.activation(x + residual)
        return x


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            activation: nn.Module = nn.ReLU(),
            norm_type: str = 'BN',
    ):
        """
        Overview:
            标准的3*3卷积构成的Residual block
        Args:
            in_channels:
            activation:
            norm_type:
        """
        super(ResBlock, self).__init__()
        self.activation = activation
        self.model_list = []
        self.conv1 = conv2d_block(in_channels, in_channels, 3, 1, activation, pad_type='zero', pad_size=1,
                                  norm_type=norm_type)
        self.conv2 = conv2d_block(in_channels, in_channels, 3, 1, None, pad_type='zero', pad_size=1,
                                  norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x + residual)
        return x
