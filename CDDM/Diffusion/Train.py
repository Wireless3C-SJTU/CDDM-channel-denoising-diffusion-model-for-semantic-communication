import math

import numpy as np
import pymongo
import torch.optim as optim
from tqdm import tqdm
import copy
from Autoencoder.data.datasets import get_loader
from Autoencoder.loss.distortion import *
from Autoencoder.net import channel, network
from Diffusion import ChannelDiffusionTrainer, ChannelDiffusionSampler
# from Diffusion.Autoencoder import AE
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from torchvision.utils import save_image
import torch.nn as nn


def train_CHDDIM(config, CHDDIM_config):
    encoder = network.JSCC_encoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    pass_channel = channel.Channel(config)
    encoder.eval()
    CDDM_config=copy.deepcopy(config)
    CDDM_config.batch_size=config.CDDM_batch
    trainLoader, _ = get_loader(CDDM_config)

    CHDDIM = UNet(T=CHDDIM_config.T, ch=CHDDIM_config.channel, ch_mult=CHDDIM_config.channel_mult,
                  attn=CHDDIM_config.attn,
                  num_res_blocks=CHDDIM_config.num_res_blocks, dropout=CHDDIM_config.dropout,
                  input_channel=CHDDIM_config.C).cuda()

    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    #
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)

    encoder.load_state_dict(torch.load(encoder_path))
    # if CHDDIM_config.training_load_weight is not None:
    #     ckpt = torch.load(CHDDIM_config.save_weight_dir + CHDDIM_config.training_load_weight)
    #     CHDDIM.load_state_dict(ckpt)

    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    #
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    optimizer = torch.optim.AdamW(
        CHDDIM.parameters(), lr=CHDDIM_config.lr, weight_decay=1e-4)
    # print(CHDDIM_config.lr)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=CHDDIM_config.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=CHDDIM_config.multiplier, warm_epoch=0.1,  # CHDDIM_config.epoch // 10,
        after_scheduler=cosineScheduler)

    trainer = ChannelDiffusionTrainer(model=CHDDIM,noise_schedule=CHDDIM_config.noise_schedule, re_weight=CHDDIM_config.re_weight,beta_1=CHDDIM_config.snr_max, beta_T=CHDDIM_config.snr_min,
                                      T=CHDDIM_config.T).cuda()

    # start training
    all_loss=[]
    
    for e in range(CHDDIM_config.epoch):
        ave_loss=0
        with tqdm(trainLoader, dynamic_ncols=True) as tqdmDataLoader:

            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.cuda()
                feature, _ = encoder(x_0)
                y = feature
                # print(y.shape)
                # print("y:",y)
                # print("mean:",feature.mode())

                y, pwr = pass_channel.complex_normalize(y, power=1)  # normalize
                if config.channel_type == "rayleigh":
                    _, h = pass_channel.reyleigh_layer(y)
                elif config.channel_type == 'awgn':
                    h = torch.ones(y.shape).cuda()
                else:
                    raise ValueError

                loss = trainer(y, h, config.SNRs, channel_type=config.channel_type)
                loss.backward()
                ave_loss+=loss
                torch.nn.utils.clip_grad_norm_(
                    CHDDIM.parameters(), CHDDIM_config.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "state": 'train_CDDM'+'re_weight'+str(CHDDIM_config.re_weight),
                    "loss: ": loss.item(),
                    "noise_schedule":CHDDIM_config.noise_schedule,
                    "input shape: ": x_0.shape,
                    "CBR": feature.numel() / 2 / x_0.numel(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        warmUpScheduler.step()
        if (e + 1) % CHDDIM_config.save_model_freq == 0:
            torch.save(CHDDIM.state_dict(), CHDDIM_config.save_path)
        all_loss.append(ave_loss.item()/50)
    #print(all_loss)

def eval_JSCC_with_CDDM(config, CHDDIM_config):
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.re_decoder_path

    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    _, testLoader = get_loader(config)

    CHDDIM = UNet(T=CHDDIM_config.T, ch=CHDDIM_config.channel, ch_mult=CHDDIM_config.channel_mult,
                  attn=CHDDIM_config.attn,
                  num_res_blocks=CHDDIM_config.num_res_blocks, dropout=CHDDIM_config.dropout,
                  input_channel=CHDDIM_config.C).cuda()

    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    encoder.load_state_dict(torch.load(encoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    CHDDIM.load_state_dict(ckpt)
    CHDDIM.eval()
    decoder.load_state_dict(torch.load(decoder_path))
    sampler = ChannelDiffusionSampler(model=CHDDIM, noise_schedule=CHDDIM_config.noise_schedule,t_max=CHDDIM_config.t_max,beta_1=CHDDIM_config.snr_max, beta_T=CHDDIM_config.snr_min,
                                      T=CHDDIM_config.T).cuda()
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    # start training
    snr_in = config.SNRs - CHDDIM_config.large_snr
    matric_aver = 0
    mse1_aver = 0
    mse2_aver = 0
    # sigma_eps_aver=torch.zeros()
    with tqdm(testLoader, dynamic_ncols=True) as tqdmtestLoader:

        for i, (images, labels) in enumerate(tqdmtestLoader):
            # train

            x_0 = images.cuda()
            feature, _ = encoder(x_0)

            y = feature
            y_0 = y
            y, pwr, h = pass_channel.forward(y, snr_in)  # normalize
            sigma_square = 1.0 / (2 * 10 ** (snr_in / 10))
            if config.channel_type == "awgn":
                y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
            elif config.channel_type == 'rayleigh':
                y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
            else:
                raise ValueError

            y = y / math.sqrt(1 + sigma_square)  # 这里可能改一下
            feature_hat = sampler(y, snr_in, snr_in + CHDDIM_config.large_snr, h, config.channel_type)

            mse2 = torch.nn.MSELoss()(feature_hat * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))

            feature_hat = feature_hat * torch.sqrt(pwr)
            x_0_hat = decoder(feature_hat)

            # optimizer1.step()
            # optimizer2.step()
            if config.loss_function == "MSE":
                mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)

                psnr = 10 * math.log10(255. * 255. / mse.item())
                matric = psnr
                #save_image(x_0_hat,"/home/wutong/semdif_revise/DIV2K_JSCCCDDM_rayleigh_PSNR_10dB/{}.png".format(i))
            elif config.loss_function == "MSSSIM":
                msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                matric = msssim
                #save_image(x_0_hat,"/home/wutong/semdif_revise/DIV2K_JSCCCDDM_rayleigh_MSSSIM_10dB/{}.png".format(i))

            mse1_aver += mse1.item()
            mse2_aver += mse2.item()
            matric_aver += matric
            CBR = feature.numel() / 2 / x_0.numel()
            tqdmtestLoader.set_postfix(ordered_dict={
                "dataset": config.dataset,
                "re_weight":str(CHDDIM_config.re_weight),
                "state": 'eval JSCC with CDDM' + config.loss_function,
                "channel": config.channel_type,
                "noise_schedule":CHDDIM_config.noise_schedule,
                "CBR": CBR,
                "SNR": snr_in,
                "matric ": matric,
                "MSE_channel": mse1.item(),
                "MSE_channel+CDDM": mse2.item(),
                "T_max":CHDDIM_config.t_max
            })
        mse1_aver = (mse1_aver / (i + 1))
        mse2_aver = (mse2_aver / (i + 1))
        matric_aver = (matric_aver / (i + 1))

        if config.loss_function == "MSE":
            name = 'PSNR'
        elif config.loss_function == "MSSSIM":
            name = "MSSSIM"
        else:
            raise ValueError
        
        #print("matric:{}",matric_aver)

        myclient = pymongo.MongoClient(config.database_address)
        mydb = myclient[config.dataset]
        if 'SNRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb['MSE' + name + '_' + config.channel_type + '_SNRs_' + 'JSCC' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, 'MSE': mse1_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb["MSE" + name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, 'MSE': mse2_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

        elif 'CBRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb['MSE' + name + '_' + config.channel_type + '_CBRs_' + 'JSCC' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, 'MSE': mse1_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb["MSE" + name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, 'MSE': mse2_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

        else:
            raise ValueError

    # print(psnr_aver/100,mse1_aver/100,mse2_aver/100)
    # eval_psnr = np.array(sigma_eps_aver / 100)
    # print(PSNR_all)
    # # print(eval_psnr.shape)
    # file = ('./CDESC_sigma_eps_rayleigh_decoderSNR5.csv'.format(CHDDIM_config.train_snr))
    # data = pd.DataFrame(eval_psnr)
    # data.to_csv(file, index=False)
    
    # eval_psnr = np.array(PSNR_all)
    # print(eval_psnr)
    # # print(eval_psnr.shape)
    # file = ('./CDESC_PSNR_rayleigh_decoderSNR5.csv'.format(CHDDIM_config.train_snr))
    # data = pd.DataFrame(eval_psnr)
    # data.to_csv(file, index=False)

def eval_JSCC_with_CDDM_SNRs(config, CHDDIM_config):
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.re_decoder_path

    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    _, testLoader = get_loader(config)

    CHDDIM = UNet(T=CHDDIM_config.T, ch=CHDDIM_config.channel, ch_mult=CHDDIM_config.channel_mult,
                  attn=CHDDIM_config.attn,
                  num_res_blocks=CHDDIM_config.num_res_blocks, dropout=CHDDIM_config.dropout,
                  input_channel=CHDDIM_config.C).cuda()

    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    encoder.load_state_dict(torch.load(encoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    CHDDIM.load_state_dict(ckpt)
    CHDDIM.eval()
    decoder.load_state_dict(torch.load(decoder_path))
    sampler = ChannelDiffusionSampler(model=CHDDIM, beta_1=CHDDIM_config.snr_max, beta_T=CHDDIM_config.snr_min,
                                      T=CHDDIM_config.T).cuda()
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    # start training
    #snr_in = config.SNRs - 3
    matric_all = []

    # sigma_eps_aver=torch.zeros()
    for snr_in in config.all_SNRs:
        matric_aver = 0

        with tqdm(testLoader, dynamic_ncols=True) as tqdmtestLoader:

            for i, (images, labels) in enumerate(tqdmtestLoader):
                # train

                x_0 = images.cuda()
                feature, _ = encoder(x_0)

                y = feature
                y, pwr, h = pass_channel.forward(y, snr_in)  # normalize
                sigma_square = 1.0 / (2 * 10 ** (snr_in / 10))

                y = y / math.sqrt(1 + sigma_square)  # 这里可能改一下
                feature_hat = sampler(y, snr_in, config.SNRs, h, config.channel_type)
                feature_hat = feature_hat * torch.sqrt(pwr)
                x_0_hat = decoder(feature_hat)

                # optimizer1.step()
                # optimizer2.step()
                if config.loss_function == "MSE":
                    mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)

                    psnr = 10 * math.log10(255. * 255. / mse.item())
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                    matric = msssim

                matric_aver += matric
                CBR = feature.numel() / 2 / x_0.numel()
                tqdmtestLoader.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": 'eval JSCC with CDDM for all SNRs' + config.loss_function,
                    "channel": config.channel_type,
                    "CBR": CBR,
                    "SNR": snr_in,
                    "matric ": matric,

                })
            matric_aver = (matric_aver / (i + 1))
            matric_all.append(matric_aver)
    if config.loss_function == "MSE":
        name = 'PSNR'
    elif config.loss_function == "MSSSIM":
        name = "MSSSIM"
    else:
        raise ValueError

    myclient = pymongo.MongoClient(config.database_address)
    mydb = myclient[config.dataset]
    mycol = mydb[name + '_' + config.channel_type + '_allSNRs_' + 'JSCC+CDDM' +'_CBR_' + str(CBR)]
    mydic = {'SNR': config.all_SNRs, name: matric_all}
    mycol.insert_one(mydic)
    print('writing successfully', mydic)

def eval_JSCC_SNRs(config):
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path

    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    _, testLoader = get_loader(config)


    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    # start training
    #snr_in = config.SNRs - 3
    matric_all = []

    # sigma_eps_aver=torch.zeros()
    for snr_in in config.all_SNRs:
        matric_aver = 0

        with tqdm(testLoader, dynamic_ncols=True) as tqdmtestLoader:

            for i, (images, labels) in enumerate(tqdmtestLoader):
                # train

                x_0 = images.cuda()
                feature, _ = encoder(x_0)

                y = feature
                y, pwr, h = pass_channel.forward(y, snr_in)  # normalize
                sigma_square_fix = 1.0 / (10 ** (snr_in / 10))
                if config.channel_type=='rayleigh':
                    y = y * (torch.conj(h)) / (torch.abs(h) ** 2 + sigma_square_fix)
                elif config.channel_type=='awgn':
                    y=y
                else:
                    raise ValueError
                y = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                feature_hat = y * torch.sqrt(pwr)
                x_0_hat = decoder(feature_hat)


                if config.loss_function == "MSE":
                    mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)

                    psnr = 10 * math.log10(255. * 255. / mse.item())
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                    matric = msssim

                matric_aver += matric
                CBR = feature.numel() / 2 / x_0.numel()
                tqdmtestLoader.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": 'eval JSCC for all SNRs' + config.loss_function,
                    "channel": config.channel_type,
                    "CBR": CBR,
                    "SNR": snr_in,
                    "matric ": matric,

                })
            matric_aver = (matric_aver / (i + 1))
            matric_all.append(matric_aver)
    if config.loss_function == "MSE":
        name = 'PSNR'
    elif config.loss_function == "MSSSIM":
        name = "MSSSIM"
    else:
        raise ValueError

    myclient = pymongo.MongoClient(config.database_address)
    mydb = myclient[config.dataset]
    mycol = mydb[name + '_' + config.channel_type + '_allSNRs_' + 'JSCC' +'_CBR_' + str(CBR)]
    mydic = {'SNR': config.all_SNRs, name: matric_all}
    mycol.insert_one(mydic)
    print('writing successfully', mydic)



def eval_JSCC_with_CDDM_delte_h(config, CHDDIM_config):
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.re_decoder_path

    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    _, testLoader = get_loader(config)

    CHDDIM = UNet(T=CHDDIM_config.T, ch=CHDDIM_config.channel, ch_mult=CHDDIM_config.channel_mult,
                  attn=CHDDIM_config.attn,
                  num_res_blocks=CHDDIM_config.num_res_blocks, dropout=CHDDIM_config.dropout,
                  input_channel=CHDDIM_config.C).cuda()

    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    encoder.load_state_dict(torch.load(encoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    CHDDIM.load_state_dict(ckpt)
    CHDDIM.eval()
    decoder.load_state_dict(torch.load(decoder_path))
    sampler = ChannelDiffusionSampler(model=CHDDIM, beta_1=CHDDIM_config.snr_max, beta_T=CHDDIM_config.snr_min,
                                      T=CHDDIM_config.T).cuda()
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    # start training
    snr_in = config.SNRs - 3
    for h_sigma in config.h_sigma:
        matric_aver = 0
        mse1_aver = 0
        mse2_aver = 0
        # sigma_eps_aver=torch.zeros()
        with tqdm(testLoader, dynamic_ncols=True) as tqdmtestLoader:

            for i, (images, labels) in enumerate(tqdmtestLoader):
                # train

                x_0 = images.cuda()
                feature, _ = encoder(x_0)

                y = feature
                y_0 = y
                y, pwr, h = pass_channel.forward(y, snr_in)  # normalize

                delte_h = h_sigma * (
                        torch.normal(mean=0.0, std=1, size=np.shape(h)) + 1j * torch.normal(mean=0.0, std=1,
                                                                                            size=np.shape(
                                                                                                h))) / np.sqrt(2)
                h = h + delte_h.cuda()

                sigma_square = 1.0 / (2 * 10 ** (snr_in / 10))
                if config.channel_type == "awgn":
                    y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                    mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
                elif config.channel_type == 'rayleigh':
                    y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                    y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                    mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
                else:
                    raise ValueError

                y = y / math.sqrt(1 + sigma_square)  # 这里可能改一下
                feature_hat = sampler(y, snr_in, snr_in + 3, h, config.channel_type)

                mse2 = torch.nn.MSELoss()(feature_hat * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))

                feature_hat = feature_hat * torch.sqrt(pwr)
                x_0_hat = decoder(feature_hat)

                # optimizer1.step()
                # optimizer2.step()
                if config.loss_function == "MSE":
                    mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)

                    psnr = 10 * math.log10(255. * 255. / mse.item())
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                    matric = msssim

                mse1_aver += mse1.item()
                mse2_aver += mse2.item()
                matric_aver += matric
                CBR = feature.numel() / 2 / x_0.numel()
                tqdmtestLoader.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": 'eval delte h JSCC with CDDM_' + config.loss_function,
                    "h sigma": h_sigma,
                    "channel": config.channel_type,
                    "CBR": CBR,
                    "SNR": snr_in,
                    "matric ": matric,
                    "MSE_channel": mse1.item(),
                    "MSE_channel+CDDM": mse2.item()
                })
            mse1_aver = (mse1_aver / (i + 1))
            mse2_aver = (mse2_aver / (i + 1))
            matric_aver = (matric_aver / (i + 1))

            if config.loss_function == "MSE":
                name = 'PSNR'
            elif config.loss_function == "MSSSIM":
                name = "MSSSIM"
            else:
                raise ValueError

            myclient = pymongo.MongoClient(config.database_address)
            mydb = myclient[config.dataset]
            if 'SNRs' in config.encoder_path:
                mycol = mydb[name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_h_sigma_' + str(
                    h_sigma) + '_CBR_' + str(CBR)]
                mydic = {'SNR': snr_in, name: matric_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)

                mycol = mydb['MSE' + name + '_' + config.channel_type + '_SNRs_' + 'JSCC' + '_h_sigma_' + str(
                    h_sigma) + '_CBR_' + str(CBR)]
                mydic = {'SNR': snr_in, 'MSE': mse1_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)

                mycol = mydb["MSE" + name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_h_sigma_' + str(
                    h_sigma) + '_CBR_' + str(CBR)]
                mydic = {'SNR': snr_in, 'MSE': mse2_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)

            elif 'CBRs' in config.encoder_path:
                mycol = mydb[name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_h_sigma_' + str(
                    h_sigma) + '_SNR_' + str(snr_in)]
                mydic = {'CBR': CBR, name: matric_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)

                mycol = mydb['MSE' + name + '_' + config.channel_type + '_CBRs_' + 'JSCC' + '_h_sigma_' + str(
                    h_sigma) + '_SNR_' + str(snr_in)]
                mydic = {'CBR': CBR, 'MSE': mse1_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)

                mycol = mydb["MSE" + name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_h_sigma_' + str(
                    h_sigma) + '_SNR_' + str(snr_in)]
                mydic = {'CBR': CBR, 'MSE': mse2_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)

            else:
                raise ValueError


def train_JSCC_with_CDDM(config, CHDDIM_config):
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path

    pass_channel = channel.Channel(config)

    trainLoader, _ = get_loader(config)
    encoder.eval()
    CHDDIM = UNet(T=CHDDIM_config.T, ch=CHDDIM_config.channel, ch_mult=CHDDIM_config.channel_mult,
                  attn=CHDDIM_config.attn,
                  num_res_blocks=CHDDIM_config.num_res_blocks, dropout=CHDDIM_config.dropout,
                  input_channel=CHDDIM_config.C).cuda()
    
    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    CHDDIM.load_state_dict(ckpt)
    CHDDIM.eval()

    sampler = ChannelDiffusionSampler(model=CHDDIM, noise_schedule=CHDDIM_config.noise_schedule,t_max=CHDDIM_config.t_max, beta_1=CHDDIM_config.snr_max, beta_T=CHDDIM_config.snr_min,
                                      T=CHDDIM_config.T).cuda()
    # optimizer_encoder = torch.optim.AdamW(
    #   encoder.parameters(), lr=CHDDIM_config.lr, weight_decay=1e-4)
    optimizer_decoder = torch.optim.AdamW(
        decoder.parameters(), lr=CHDDIM_config.lr, weight_decay=1e-4)

    # start training
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    for e in range(config.retrain_epoch):
        with tqdm(trainLoader, dynamic_ncols=True) as tqdmtrainLoader:

            for i, (images, labels) in enumerate(tqdmtrainLoader):
                # train
                snr = config.SNRs - CHDDIM_config.large_snr

                x_0 = images.cuda()
                feature, _ = encoder(x_0)
                y = feature

                y, pwr, h = pass_channel.forward(y, snr)  # normalize
                sigma_square = 1.0 / (2 * 10 ** (snr / 10))
                y = y / math.sqrt(1 + sigma_square)  # 这里可能改一下
                feature_hat = sampler(y, snr, snr + CHDDIM_config.large_snr, h, config.channel_type)

                feature_hat = feature_hat * torch.sqrt(pwr)
                x_0_hat = decoder(feature_hat)

                # mse1=torch.nn.MSEloss()()
                if config.loss_function == "MSE":
                    loss = torch.nn.MSELoss()(x_0, x_0_hat)
                elif config.loss_function == "MSSSIM":
                    loss = CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean()
                else:
                    raise ValueError

                optimizer_decoder.zero_grad()
                loss.backward()
                optimizer_decoder.step()
                # optimizer_encoder.step()
                if config.loss_function == "MSE":
                    mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)
                    psnr = 10 * math.log10(255. * 255. / mse.item())
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                    matric = msssim

                tqdmtrainLoader.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": "train_decoder" + config.loss_function,
                    "noise_schedule":CHDDIM_config.noise_schedule,
                    "channel": config.channel_type,
                    "CBR:": feature.numel() / 2 / x_0.numel(),
                    "SNR": snr,
                    "matric": matric,
                    "T_max":CHDDIM_config.t_max
                })

            if (e + 1) % config.retrain_save_model_freq == 0:
                torch.save(decoder.state_dict(), config.re_decoder_path)

def train_DnCNN(config,CHDDIM_config):
    from DnCNN.models import DnCNN
    import torch.nn as nn
    encoder = network.JSCC_encoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    pass_channel = channel.Channel(config)
    encoder.eval()
    CNN_config=copy.deepcopy(config)
    CNN_config.batch_size=config.CDDM_batch
    trainLoader, _ = get_loader(CNN_config)

    DeCNN=DnCNN(config.C).cuda()
    DeCNN.train()
    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    #
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)

    encoder.load_state_dict(torch.load(encoder_path))
    # if CHDDIM_config.training_load_weight is not None:
    #     ckpt = torch.load(CHDDIM_config.save_weight_dir + CHDDIM_config.training_load_weight)
    #     CHDDIM.load_state_dict(ckpt)

    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    #
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    optimizer = torch.optim.Adam(
        DeCNN.parameters(), lr=CNN_config.learning_rate)
    # print(CHDDIM_config.lr)
    # start training
    for e in range(CHDDIM_config.epoch):
        with tqdm(trainLoader, dynamic_ncols=True) as tqdmDataLoader:

            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.cuda()
                feature, _ = encoder(x_0)
                y = feature
                # print(y.shape)
                # print("y:",y)
                # print("mean:",feature.mode())
                
                y, pwr, h = pass_channel.forward(y, config.SNRs)
                sigma_square = 1.0 / (2 * 10 ** (config.SNRs / 10))
                if config.channel_type == "awgn":
                    y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                    noise=y_awgn-feature
                    #mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_awgn
                elif config.channel_type == 'rayleigh':
                    y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                    y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                    noise=y_mmse-feature
                    #mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y* math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_mmse
                else:
                    raise ValueError
                output=DeCNN(receive)
                loss=nn.MSELoss()(output, noise)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "channel_type":config.channel_type,
                    "state": 'train_DeCNN',
                    "loss: ": loss.item(),
                    "input shape: ": x_0.shape,
                    "CBR": feature.numel() / 2 / x_0.numel(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })


        if (e + 1) % CHDDIM_config.save_model_freq == 0:
            torch.save(DeCNN.state_dict(), CHDDIM_config.save_path)

def train_JSCC_with_DnCNN(config, CHDDIM_config):
    from DnCNN.models import DnCNN
    import torch.nn as nn
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path

    pass_channel = channel.Channel(config)

    trainLoader, _ = get_loader(config)
    encoder.eval()
    DnCNN=DnCNN(config.C).cuda()

    
    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    DnCNN.load_state_dict(ckpt)
    DnCNN.eval()

    # optimizer_encoder = torch.optim.AdamW(
    #   encoder.parameters(), lr=CHDDIM_config.lr, weight_decay=1e-4)
    optimizer_decoder = torch.optim.Adam(
        decoder.parameters(), lr=CHDDIM_config.lr)

    # start training
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    for e in range(config.retrain_epoch):
        with tqdm(trainLoader, dynamic_ncols=True) as tqdmtrainLoader:

            for i, (images, labels) in enumerate(tqdmtrainLoader):
                # train
                snr = config.SNRs - CHDDIM_config.large_snr

                x_0 = images.cuda()
                feature, _ = encoder(x_0)
                y = feature

                y, pwr, h = pass_channel.forward(y, snr)  # normalize
                sigma_square = 1.0 / (2 * 10 ** (config.SNRs / 10))
                if config.channel_type == "awgn":
                    y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                    #mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_awgn
                elif config.channel_type == 'rayleigh':
                    y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                    y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                    #mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_mmse
                else:
                    raise ValueError
                feature_hat = receive-DnCNN(receive)

                feature_hat = feature_hat * torch.sqrt(pwr)
                x_0_hat = decoder(feature_hat)

                # mse1=torch.nn.MSEloss()()
                if config.loss_function == "MSE":
                    loss = torch.nn.MSELoss()(x_0, x_0_hat)
                elif config.loss_function == "MSSSIM":
                    loss = CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean()
                else:
                    raise ValueError

                optimizer_decoder.zero_grad()
                loss.backward()
                optimizer_decoder.step()
                # optimizer_encoder.step()
                if config.loss_function == "MSE":
                    mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)
                    psnr = 10 * math.log10(255. * 255. / mse.item())
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                    matric = msssim

                tqdmtrainLoader.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": "train_decoder" + config.loss_function,
                    "noise_schedule":CHDDIM_config.noise_schedule,
                    "channel": config.channel_type,
                    "CBR:": feature.numel() / 2 / x_0.numel(),
                    "SNR": snr,
                    "matric": matric,
                    "T_max":CHDDIM_config.t_max
                })

            if (e + 1) % config.retrain_save_model_freq == 0:
                torch.save(decoder.state_dict(), config.re_decoder_path)

def eval_JSCC_with_DnCNN(config, CHDDIM_config):
    from DnCNN.models import DnCNN
    import torch.nn as nn
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.re_decoder_path

    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    _, testLoader = get_loader(config)
    DnCNN=DnCNN(config.C).cuda()

    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    encoder.load_state_dict(torch.load(encoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    DnCNN.load_state_dict(ckpt)
    DnCNN.eval()
    decoder.load_state_dict(torch.load(decoder_path))

    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    # start training
    snr_in = config.SNRs - CHDDIM_config.large_snr
    matric_aver = 0
    mse1_aver = 0
    mse2_aver = 0
    # sigma_eps_aver=torch.zeros()
    with tqdm(testLoader, dynamic_ncols=True) as tqdmtestLoader:

        for i, (images, labels) in enumerate(tqdmtestLoader):
            # train

            x_0 = images.cuda()
            feature, _ = encoder(x_0)

            y = feature
            y_0 = y
            y, pwr, h = pass_channel.forward(y, snr_in)  # normalize
            sigma_square = 1.0 / (2 * 10 ** (config.SNRs / 10))
            if config.channel_type == "awgn":
                y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
                receive=y_awgn
            elif config.channel_type == 'rayleigh':
                y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
                receive=y_mmse
            else:
                raise ValueError

            
            feature_hat = receive-DnCNN(receive)

            mse2 = torch.nn.MSELoss()(feature_hat * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))

            feature_hat = feature_hat * torch.sqrt(pwr)
            x_0_hat = decoder(feature_hat)

            # optimizer1.step()
            # optimizer2.step()
            if config.loss_function == "MSE":
                mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)

                psnr = 10 * math.log10(255. * 255. / mse.item())
                matric = psnr
                #save_image(x_0_hat,"/home/wutong/semdif_revise/DIV2K_JSCCCDDM_rayleigh_PSNR_10dB/{}.png".format(i))
            elif config.loss_function == "MSSSIM":
                msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                matric = msssim
                #save_image(x_0_hat,"/home/wutong/semdif_revise/DIV2K_JSCCCDDM_rayleigh_MSSSIM_10dB/{}.png".format(i))

            mse1_aver += mse1.item()
            mse2_aver += mse2.item()
            matric_aver += matric
            CBR = feature.numel() / 2 / x_0.numel()
            tqdmtestLoader.set_postfix(ordered_dict={
                "dataset": config.dataset,
                "re_weight":str(CHDDIM_config.re_weight),
                "state": 'eval JSCC with CDDM' + config.loss_function,
                "channel": config.channel_type,
                "noise_schedule":CHDDIM_config.noise_schedule,
                "CBR": CBR,
                "SNR": snr_in,
                "matric ": matric,
                "MSE_channel": mse1.item(),
                "MSE_channel+CDDM": mse2.item(),
                "T_max":CHDDIM_config.t_max
            })
        mse1_aver = (mse1_aver / (i + 1))
        mse2_aver = (mse2_aver / (i + 1))
        matric_aver = (matric_aver / (i + 1))

        if config.loss_function == "MSE":
            name = 'PSNR'
        elif config.loss_function == "MSSSIM":
            name = "MSSSIM"
        else:
            raise ValueError
        
        #print("matric:{}",matric_aver)

        myclient = pymongo.MongoClient(config.database_address)
        mydb = myclient[config.dataset]
        if 'SNRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb['MSE' + name + '_' + config.channel_type + '_SNRs_' + 'JSCC' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, 'MSE': mse1_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb["MSE" + name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, 'MSE': mse2_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

        elif 'CBRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb['MSE' + name + '_' + config.channel_type + '_CBRs_' + 'JSCC' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, 'MSE': mse1_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb["MSE" + name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, 'MSE': mse2_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

        else:
            raise ValueError

def train_GAN(config,CHDDIM_config):
    from WGANVGG.networks import WGAN_VGG, WGAN_VGG_generator
    train_losses = []
    encoder = network.JSCC_encoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    pass_channel = channel.Channel(config)
    encoder.eval()
    GAN_config=copy.deepcopy(config)
    GAN_config.batch_size=config.CDDM_batch
    trainLoader, _ = get_loader(GAN_config)

    WGAN_VGG=WGAN_VGG(input_size=16,in_channels=config.C).cuda()
    WGAN_VGG_generator=WGAN_VGG_generator()
    
    criterion_perceptual = torch.nn.L1Loss()
    optimizer_g = torch.optim.Adam(WGAN_VGG.generator.parameters(), CHDDIM_config.lr)
    optimizer_d = torch.optim.Adam(WGAN_VGG.discriminator.parameters(), CHDDIM_config.lr)
    encoder.load_state_dict(torch.load(encoder_path))

    for e in range(CHDDIM_config.epoch):
        with tqdm(trainLoader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                snr = config.SNRs - CHDDIM_config.large_snr

                x_0 = images.cuda()
                feature, _ = encoder(x_0)
                y = feature

                y, pwr, h = pass_channel.forward(y, snr)  # normalize
                sigma_square = 1.0 / (2 * 10 ** (config.SNRs / 10))
                if config.channel_type == "awgn":
                    y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                    #mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_awgn*torch.sqrt(pwr)
                elif config.channel_type == 'rayleigh':
                    y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                    y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                    #mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_mmse*torch.sqrt(pwr)
                else:
                    raise ValueError
                
                
                for index_2 in range(GAN_config.n_d_train):
                    
                    optimizer_d.zero_grad()
                    #WGAN_VGG.discriminator.zero_grad()
                    #mse1 = torch.nn.MSELoss()(receive / torch.sqrt(pwr) * math.sqrt(2), feature * math.sqrt(2) / torch.sqrt(pwr))
                    #print(mse1.item())
                    d_loss, gp_loss = WGAN_VGG.d_loss(receive, feature, gp=True, return_gp=True)
                    d_loss.backward(retain_graph=True)
                    optimizer_d.step()

                optimizer_g.zero_grad()

                g_loss,p_loss= WGAN_VGG.g_loss(receive, feature, perceptual=True, return_p=True)
                #print(pwr)
                g_loss.backward()
                optimizer_g.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "state": 'train_GAN',
                    "channel type":config.channel_type,
                    "g loss: ": g_loss.item()-p_loss.item(), 
                    "p loss: ": p_loss.item(), 
                    "d loss: ": d_loss.item(), 
                    "d-gp loss: ":d_loss.item()-gp_loss.item(), 
                    "gp loss: ":gp_loss.item(),
                    "input shape: ": x_0.shape,
                    "CBR": feature.numel() / 2 / x_0.numel(),
                })

        if (e + 1) % CHDDIM_config.save_model_freq == 0:
            torch.save(WGAN_VGG.state_dict(), CHDDIM_config.save_path)

def eval_JSCC_with_GAN(config, CHDDIM_config):
    from WGANVGG.networks import WGAN_VGG, WGAN_VGG_generator
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.re_decoder_path

    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    _, testLoader = get_loader(config)

    WGAN_VGG=WGAN_VGG(input_size=16,in_channels=config.C).cuda()

    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])

    encoder.load_state_dict(torch.load(encoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    WGAN_VGG.load_state_dict(ckpt)
    WGAN_VGG.eval()
    decoder.load_state_dict(torch.load(decoder_path))

    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    # start training
    snr_in = config.SNRs - CHDDIM_config.large_snr
    matric_aver = 0
    mse1_aver = 0
    mse2_aver = 0
    # sigma_eps_aver=torch.zeros()
    with tqdm(testLoader, dynamic_ncols=True) as tqdmtestLoader:

        for i, (images, labels) in enumerate(tqdmtestLoader):
            # train

            x_0 = images.cuda()
            feature, _ = encoder(x_0)

            y = feature
            y_0 = y
            y, pwr, h = pass_channel.forward(y, snr_in)  # normalize
            sigma_square = 1.0 / (2 * 10 ** (snr_in / 10))
            if config.channel_type == "awgn":
                y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
                receive=y_awgn*torch.sqrt(pwr)
            elif config.channel_type == 'rayleigh':
                y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y_0 * math.sqrt(2) / torch.sqrt(pwr))
                receive=y_mmse*torch.sqrt(pwr)
            else:
                raise ValueError

            feature_hat=WGAN_VGG.generator(receive)
            mse2 = torch.nn.MSELoss()(feature_hat * math.sqrt(2)/ torch.sqrt(pwr), y_0 * math.sqrt(2) / torch.sqrt(pwr))

           # feature_hat = feature_hat * torch.sqrt(pwr)
            x_0_hat = decoder(feature_hat)

            # optimizer1.step()
            # optimizer2.step()
            if config.loss_function == "MSE":
                mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)

                psnr = 10 * math.log10(255. * 255. / mse.item())
                matric = psnr
                #save_image(x_0_hat,"/home/wutong/semdif_revise/DIV2K_JSCCCDDM_rayleigh_PSNR_10dB/{}.png".format(i))
            elif config.loss_function == "MSSSIM":
                msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                matric = msssim
                #save_image(x_0_hat,"/home/wutong/semdif_revise/DIV2K_JSCCCDDM_rayleigh_MSSSIM_10dB/{}.png".format(i))

            mse1_aver += mse1.item()
            mse2_aver += mse2.item()
            matric_aver += matric
            CBR = feature.numel() / 2 / x_0.numel()
            tqdmtestLoader.set_postfix(ordered_dict={
                "dataset": config.dataset,
                "state": 'eval JSCC with GAN' + config.loss_function,
                "channel": config.channel_type,
                "CBR": CBR,
                "SNR": snr_in,
                "matric ": matric,
                "MSE_channel": mse1.item(),
                "MSE_channel+GAN": mse2.item(),
            })
        mse1_aver = (mse1_aver / (i + 1))
        mse2_aver = (mse2_aver / (i + 1))
        matric_aver = (matric_aver / (i + 1))

        if config.loss_function == "MSE":
            name = 'PSNR'
        elif config.loss_function == "MSSSIM":
            name = "MSSSIM"
        else:
            raise ValueError
        
        #print("matric:{}",matric_aver)

        myclient = pymongo.MongoClient(config.database_address)
        mydb = myclient[config.dataset]
        if 'SNRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb['MSE' + name + '_' + config.channel_type + '_SNRs_' + 'JSCC' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, 'MSE': mse1_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb["MSE" + name + '_' + config.channel_type + '_SNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)]
            mydic = {'SNR': snr_in, 'MSE': mse2_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

        elif 'CBRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb['MSE' + name + '_' + config.channel_type + '_CBRs_' + 'JSCC' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, 'MSE': mse1_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

            mycol = mydb["MSE" + name + '_' + config.channel_type + '_CBRs_' + 'JSCC+CDDM' + '_SNR_' + str(snr_in)]
            mydic = {'CBR': CBR, 'MSE': mse2_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)

        else:
            raise ValueError

def train_JSCC_with_GAN(config, CHDDIM_config):
    from WGANVGG.networks import WGAN_VGG, WGAN_VGG_generator
    encoder = network.JSCC_encoder(config, config.C).cuda()
    decoder = network.JSCC_decoder(config, config.C).cuda()
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path

    pass_channel = channel.Channel(config)

    trainLoader, _ = get_loader(config)
    encoder.eval()
    WGAN_VGG=WGAN_VGG(input_size=16,in_channels=config.C).cuda()

    
    # encoder = torch.nn.DataParallel(encoder, device_ids=CHDDIM_config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=CHDDIM_config.device_ids)
    # CHDDIM = torch.nn.DataParallel(CHDDIM, device_ids=CHDDIM_config.device_ids)
    #
    # encoder = encoder.cuda(device=CHDDIM_config.device_ids[0])
    # decoder = decoder.cuda(device=CHDDIM_config.device_ids[0])
    # CHDDIM = CHDDIM.cuda(device=CHDDIM_config.device_ids[0])
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    ckpt = torch.load(CHDDIM_config.save_path)
    WGAN_VGG.load_state_dict(ckpt)
    WGAN_VGG.eval()

    # optimizer_encoder = torch.optim.AdamW(
    #   encoder.parameters(), lr=CHDDIM_config.lr, weight_decay=1e-4)
    optimizer_decoder = torch.optim.Adam(
        decoder.parameters(), lr=CHDDIM_config.lr)

    # start training
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    for e in range(config.retrain_epoch):
        with tqdm(trainLoader, dynamic_ncols=True) as tqdmtrainLoader:

            for i, (images, labels) in enumerate(tqdmtrainLoader):
                # train
                snr = config.SNRs - CHDDIM_config.large_snr

                x_0 = images.cuda()
                feature, _ = encoder(x_0)
                y = feature

                y, pwr, h = pass_channel.forward(y, snr)  # normalize
                sigma_square = 1.0 / (2 * 10 ** (config.SNRs / 10))
                if config.channel_type == "awgn":
                    y_awgn = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                    #mse1 = torch.nn.MSELoss()(y_awgn * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_awgn* torch.sqrt(pwr)
                elif config.channel_type == 'rayleigh':
                    y_mmse = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                    y_mmse = torch.cat((torch.real(y_mmse), torch.imag(y_mmse)), dim=2)
                    #mse1 = torch.nn.MSELoss()(y_mmse * math.sqrt(2), y * math.sqrt(2) / torch.sqrt(pwr))
                    receive=y_mmse* torch.sqrt(pwr)
                else:
                    raise ValueError
                feature_hat = WGAN_VGG.generator(receive)

                #feature_hat = feature_hat * torch.sqrt(pwr)
                x_0_hat = decoder(feature_hat)

                # mse1=torch.nn.MSEloss()()
                if config.loss_function == "MSE":
                    loss = torch.nn.MSELoss()(x_0, x_0_hat)
                elif config.loss_function == "MSSSIM":
                    loss = CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean()
                else:
                    raise ValueError

                optimizer_decoder.zero_grad()
                loss.backward()
                optimizer_decoder.step()
                # optimizer_encoder.step()
                if config.loss_function == "MSE":
                    mse = torch.nn.MSELoss()(x_0 * 255., x_0_hat.clamp(0., 1.) * 255)
                    psnr = 10 * math.log10(255. * 255. / mse.item())
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    msssim = 1 - CalcuSSIM(x_0, x_0_hat.clamp(0., 1.)).mean().item()
                    matric = msssim

                tqdmtrainLoader.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": "train_decoder",
                    "noise_schedule":"GAN",
                    "channel": config.channel_type,
                    "CBR:": feature.numel() / 2 / x_0.numel(),
                    "SNR": snr,
                    "matric": matric,
                    
                })

            if (e + 1) % config.retrain_save_model_freq == 0:
                torch.save(decoder.state_dict(), config.re_decoder_path)


class netCDDM(nn.Module):

    def __init__(self,config,CHDDIM_config):
        super().__init__()
        self.CDDM=UNet(T=CHDDIM_config.T, ch=CHDDIM_config.channel, ch_mult=CHDDIM_config.channel_mult,
                attn=CHDDIM_config.attn,
                num_res_blocks=CHDDIM_config.num_res_blocks, dropout=CHDDIM_config.dropout,
                input_channel=CHDDIM_config.C).cuda()

    def forward(self,input):
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input)) ** 2
                    + torch.normal(mean=0.0, std=1, size=np.shape(input)) ** 2) / np.sqrt(2)
        h = h.cuda()
        t = input.new_ones([input.shape[0], ], dtype=torch.long) * 100
        t=t.cuda()
        x=self.CDDM(input,t,h)
        return x
    
class netJSCC(nn.Module):

    def __init__(self,config,CHDDIM_config):
        super().__init__()
        self.encoder = network.JSCC_encoder(config, config.C).cuda()
        self.decoder = network.JSCC_decoder(config, config.C).cuda()

    def forward(self,input):
        x,_=self.encoder(input)
        y=self.decoder(x)
        return y

def test_mem_and_comp(config,CHDDIM_config):

    from thop import profile
    from thop import clever_format



    network=netJSCC(config,CHDDIM_config)
    
    input=torch.randn(1,3,256,256).cuda()
    macs,params=profile(network,inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs,params)