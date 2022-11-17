from typing import Union, Optional, Dict, Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoiseLinearLayer(nn.Module):
    r"""
    Overview:
        Linear layer with random noise.
    Interface:
        reset_noise, reset_parameters, forward
    """

    def __init__(self, in_channels: int, out_channels: int, sigma0: int = 0.4) -> None:
        super(NoiseLinearLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_channels))
        # self.register_buffer("weight_eps", torch.empty(out_channels, in_channels))
        # self.register_buffer("bias_eps", torch.empty(out_channels))
        self.sigma0 = sigma0
        self.reset_parameter()
        self.reset_noise()

    def _scale_noise(self, size: Union[int, Tuple]):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        r"""
        Overview:
            Reset noise settinngs in the layer.
        """
        is_cuda = self.weight_mu.is_cuda
        in_noise = self._scale_noise(self.in_channels).to(torch.device('cuda' if is_cuda else 'cpu'))
        out_noise = self._scale_noise(self.out_channels).to(torch.device('cuda' if is_cuda else 'cpu'))
        self.weight_eps = out_noise.ger(in_noise)
        self.bias_eps = out_noise

    def reset_parameters(self):
        r"""
        Overview:
            Reset parameters in the layer.
        """
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)

        std_weight = self.sigma0 / math.sqrt(self.in_channels)
        self.weight_sigma.data.fill_(std_weight)
        std_bias = self.sigma0 / math.sqrt(self.out_channels)
        self.bias_sigma.data.fill_(std_bias)

    def forward(self, x: torch.Tensor):
        r"""
        Overview:
            Layer forward with noise.
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Returns:
            - output (:obj:`torch.Tensor`): the output with noise
        """
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_eps,
                self.bias_mu + self.bias_sigma * self.bias_eps,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


def noise_block(
        in_channels: int,
        out_channels: int,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5,
        sigma0: int = 0.4
):
    model_list = [NoiseLinearLayer(in_channels, out_channels, sigma0=sigma0)]
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
        model_list.append(nn.Dropout(dropout_probability))
    model = nn.Sequential(*model_list)
    return model
