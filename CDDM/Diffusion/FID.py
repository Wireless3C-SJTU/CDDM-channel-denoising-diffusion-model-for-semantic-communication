#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
@file:file
@time: time
"""
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis= 0), cov(act1, rowvar= False)
    mu2, sigma2 = act2.mean(axis= 0), cov(act2, rowvar= False)

    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)* 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean)
    return fid




