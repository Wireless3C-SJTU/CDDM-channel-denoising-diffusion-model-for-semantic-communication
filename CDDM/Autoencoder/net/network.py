from random import choice

import torch.nn

# from loss.distortion import Distortion
#from Autoencoder.loss.distortion import Distortion
from Autoencoder.net.channel import Channel
from Autoencoder.net.decoder import *
from Autoencoder.net.encoder import *

class JSCC_encoder(nn.Module):
    def __init__(self, config, C):
        super(JSCC_encoder, self).__init__()
        self.config = config
        self.config.encoder_kwargs["C"] = C
        encoder_kwargs = config.encoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)


    def forward(self, input_image):
        feature = self.encoder(input_image)
        #CBR = feature.module.num.numel() / 2 / input_image.numel()
        return feature,1

class JSCC_decoder(nn.Module):
    def __init__(self, config,C):
        super(JSCC_decoder, self).__init__()
        self.config = config
        self.config.decoder_kwargs["C"]=C
        decoder_kwargs = config.decoder_kwargs
        self.decoder = create_decoder(**decoder_kwargs)


    def forward(self, feature):
        recon_image = self.decoder(feature)

        return recon_image
