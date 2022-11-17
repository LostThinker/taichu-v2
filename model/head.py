from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn
from model.noise import noise_block
from model.block import fc_block


class DiscreteHead(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            noise: Optional[bool] = False,
            noise_sigma0: Optional[int] = 0.4
    ):
        super(DiscreteHead, self).__init__()
        model_list = []
        if noise:
            for i in range(layer_num - 1):
                model_list.append(noise_block(input_size, input_size, activation, norm_type, sigma0=noise_sigma0))
            model_list.append(noise_block(input_size, output_size, None, norm_type, sigma0=noise_sigma0))
        else:
            for i in range(layer_num - 1):
                model_list.append(fc_block(input_size, input_size, activation, norm_type))
            model_list.append(fc_block(input_size, output_size, None, norm_type))

        self.model = nn.Sequential(*model_list)

    def forward(self, x: torch.Tensor):
        logit = self.model(x)
        return {'logit': logit}


class DistributionHead(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_num: int = 1,
            n_atom: int = 51,
            v_min: float = -10,
            v_max: float = 10,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            noise: Optional[bool] = False,
            noise_sigma0: Optional[int] = 0.4,
            eps: Optional[float] = 1e-6,
    ) -> None:
        super(DistributionHead, self).__init__()
        model_list = []

        if noise:
            for i in range(layer_num - 1):
                model_list.append(noise_block(input_size, input_size, activation, norm_type, sigma0=noise_sigma0))
            model_list.append(noise_block(input_size, output_size, activation, norm_type, sigma0=noise_sigma0))
        else:
            for i in range(layer_num - 1):
                model_list.append(fc_block(input_size, input_size, activation, norm_type))
            model_list.append(fc_block(input_size, output_size, None, norm_type))
        self.model = nn.Sequential(*model_list)

        self.output_size = output_size
        self.n_atom = n_atom
        self.v_min = v_min
        self.v_max = v_max
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Dict:
        q = self.model(x)
        q = q.view(*q.shape[:-1], self.output_size, self.n_atom)
        dist = torch.softmax(q, dim=-1) + self.eps
        q = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(x)
        q = q.sum(-1)
        return {'logit': q, 'distribution': dist}
