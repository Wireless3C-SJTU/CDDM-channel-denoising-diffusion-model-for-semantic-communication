import numpy as np
import torch
import torch.nn as nn


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self,config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel_type
        self.device = config.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414


    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.get_device()

        # print(np.shape(input_layer))
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = (torch.normal(mean=0.0, std=1, size=np.shape(input_layer))
             + 1j * torch.normal(mean=0.0, std=1, size=np.shape(input_layer))) / np.sqrt(2)
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
            h = h.to(input_layer.get_device())
        return input_layer * h + noise, h

    def complex_normalize(self, x, power):
        # print(x.shape)
        pwr = torch.mean(x ** 2) * 2  # 复数功率是实数功率2倍
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def reyleigh_layer(self, x):

        L = x.shape[2]
        channel_in = x[:, :, :L // 2, :] + x[:, :, L // 2:, :] * 1j
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(channel_in)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(channel_in)) ** 2) / np.sqrt(2)
        h = h.cuda()
        channel_output = channel_in * h
        channel_output = torch.cat((torch.real(channel_output), torch.imag(channel_output)), dim=2)
        # channel_output = channel_output.reshape(x.shape)
        # h = torch.cat((torch.real(h), torch.imag(h)), dim=2)
        # h = h.reshape(x.shape)

        return channel_output, h

    def forward(self, input, chan_param, avg_pwr=False):
        if avg_pwr:
            power = 1
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        # print(input.shape)
        input_shape = channel_tx.shape
        # channel_in = channel_tx.reshape(-1)
        channel_in = channel_tx
        L = channel_in.shape[2]
        channel_in = channel_in[:, :, :L // 2, :] + channel_in[:, :, L // 2:, :] * 1j
        channel_output, h = self.complex_forward(channel_in, chan_param)
        #channel_output = torch.cat((torch.real(channel_output), torch.imag(channel_output)), dim=2)
        # h = torch.cat((torch.real(h), torch.imag(h)), dim=2)
        # channel_output = channel_output.reshape(input_shape)
        if self.chan_type == 1 or self.chan_type == 'awgn':
            #noise = (channel_output - channel_tx).detach()
            #noise.requires_grad = False
            #channel_tx = channel_tx + noise
            return channel_output, pwr,torch.ones(channel_output.shape).cuda()
            # if avg_pwr:
            #     return channel_tx * torch.sqrt(avg_pwr * 2)
            # else:
            #     return channel_tx * torch.sqrt(pwr)

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            
            return channel_output, pwr, h

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))  # 实部虚部分别加，所以除2
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output,1

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output, h = self.rayleigh_noise_layer(channel_tx,
                                                       std=sigma,
                                                       name="rayleigh_chan_noise")
            return chan_output, h

    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx
