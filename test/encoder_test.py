from model.encoder import FCEncoder, ConvEncoder
import torch


def test():
    FCEncoder_test()
    ConvEncoder_test()


def FCEncoder_test():
    obs_shape = 10
    hidden_size_list = [16, 32, 12]
    model = FCEncoder(obs_shape, hidden_size_list)
    data = torch.randn(5, 10)
    print('input shape', data.shape)
    out = model(data)
    print('output shape', out.shape)


def ConvEncoder_test():
    obs_shape = [3, 84, 84]
    output_shape = 10
    hidden_channel_list = [8, 16, 16]
    kernel_size_list = [3, 3, 3]
    stride_list = [2, 1, 1]
    model = ConvEncoder(obs_shape, output_shape, hidden_channel_list, kernel_size_list, stride_list)
    data = torch.randn(10, 3, 84, 84)
    print('input shape', data.shape)
    out = model(data)
    print('output shape', out.shape)


if __name__ == '__main__':
    test()
