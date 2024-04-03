#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
@file:file
@time: time
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import pandas as pd
# from Diffusion.Autoencoder import noise_encoder
from torch.autograd import Variable
from torchvision.utils import save_image
import pymongo

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        # self.AE.train()

        # self.SNR = torch.linspace(beta_1, beta_T, T)
        # self.betas = 10 ** (-self.SNR / 10.)
        # self.sqrt_betas=torch.sqrt(self.betas)
        # self.SNR = -10 * torch.log10(self.betas)
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())

        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T]
        #self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)
        self.pow = self.alphas_bar / (1. - self.alphas_bar)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        #self.register_buffer(
        #     'sqrt_one_minus_alphas_bar', torch.sqrt(1. - self.alphas_bar))
        self.register_buffer('SNR',(1-self.alphas_bar)/self.alphas_bar)

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - self.alphas_bar))
        self.register_buffer('coeff3', self.coeff1 * (1. - alphas))

        self.register_buffer('posterior_var', self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar))
        self.register_buffer('sigma_eps',(alphas-torch.sqrt(alphas-self.alphas_bar))/(torch.sqrt(1-self.alphas_bar)-torch.sqrt(alphas-self.alphas_bar)))

# sampler = GaussianDiffusionSampler(model=1, beta_1=1e-4, beta_T=0.02, T=1000)
# #data=pd.read_csv(r"E:\code\DDPM\semdif\CDESC_sigma_eps_rayleigh_decoderSNR5.csv")
# #print(sampler.SNR)
# #plt.plot(sampler.sigma_eps)
# #plt.show()
# sigma_eps=[]
# for i in range(999):
#     t=i+1
#     x=1-sampler.alphas_bar[t-1]-torch.sqrt(1/(1-sampler.betas[t])-sampler.alphas_bar[t-1])
#     y=(1-sampler.alphas_bar[t])-torch.sqrt((1-sampler.alphas_bar[t-1])*(1-sampler.alphas_bar[t])/(1-sampler.betas[t]))
#     sigma_eps.append(y/x)
# plt.plot(sigma_eps)
# plt.show()
#120*5 -- 1
#124*5 -- 0.5
# print(sampler.betas)
mycilent = pymongo.MongoClient("mongodb://localhost:27017")
print(mycilent)
mydb = mycilent["test"]
mycol=mydb["test_col"]
mydic={'1':3}
mycol.insert_one(mydic)

#print(list(mycol.find({})[0]))

