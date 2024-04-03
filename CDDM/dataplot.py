#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
@file:file
@time: time
"""

import matplotlib.pyplot as plt
import numpy as np
import pymongo


def sortvalue(val):
    if val.get('SNR') != None:
        return val.get('SNR')
    elif val.get('CBR') != None:
        return val.get('CBR')
    else:
        raise ValueError


def acquire_data(config, name, data_name):
    mycilent = pymongo.MongoClient(config.database_address)
    mydb = mycilent[config.dataset]
    mycol = mydb[name]
    assert data_name in [0, 1, 2]
    datas = list(mycol.find({}))
    datas.sort(key=sortvalue)
    snr = []
    msssim = []
    if 'SNRs' in name:
        if data_name == 0:
            for data in datas:
                snr.append(data.get('SNR'))
                msssim.append(data.get('PSNR'))
            return snr, msssim
        elif data_name == 1:
            for data in datas:
                snr.append(data.get('SNR'))
                msssim.append(-10 * np.log10(1 - data.get('MSSSIM')))
            return snr, msssim
        elif data_name == 2:
            for data in datas:
                snr.append(data.get('SNR'))
                msssim.append(data.get('MSE'))
            return snr, msssim
        else:
            raise ValueError
    elif 'CBRs' in name:
        if data_name == 0:
            for data in datas:
                snr.append(data.get('CBR'))
                msssim.append(data.get('PSNR'))
            return snr, msssim
        elif data_name == 1:
            for data in datas:
                snr.append(data.get('CBR'))
                msssim.append(-10 * np.log10(1 - data.get('MSSSIM')))
            return snr, msssim
        elif data_name == 2:
            for data in datas:
                snr.append(data.get('CBR'))
                msssim.append(data.get('MSE'))
            return snr, msssim
        else:
            raise ValueError
    else:
        raise ValueError


def plot_matric(config):
    plt.figure(figsize=(15, 8))
    if config.loss_function == "MSE":
        name = 'PSNR'
        data_name = 0
    elif config.loss_function == "MSSSIM":
        name = "MSSSIM"
        data_name = 1
    else:
        raise ValueError

    if config.dataset == "CIFAR10":
        CBR = config.C / (12 * 8)
    elif config.dataset == "DIV2K":
        CBR = config.C / (96 * 12)
    elif config.dataset == "CelebA":
        CBR = config.C / (24 * 16)

    if 'SNRs' in config.encoder_path:
        fix = '_CBR_' + str(CBR)
        type = '_SNRs_'
        data_address = name + '_' + config.channel_type + '_allSNRs_' + 'JSCC+CDDM' + '_CBR_' + str(CBR)
        mycilent = pymongo.MongoClient(config.database_address)
        mydb = mycilent[config.dataset]
        mycol = mydb[data_address]
        data = list(mycol.find({}))[0]
        snrs = data.get('SNR')
        text = 'JSCC+CDDM train at SNR=' + str(config.all_SNRs[0]) + ' (dB)'
        if data_name == 1:
            msssims = [-10 * np.log10(1 - data.get('MSSSIM')[i]) for i in range(len(data.get('MSSSIM')))]
            plt.plot(snrs, msssims, label=text, marker='.', markersize=10)
        elif data_name == 0:
            psnrs = data.get("PSNR")
            plt.plot(snrs, psnrs, label=text, marker='.', markersize=10)
        else:
            raise ValueError
    elif 'CBRs' in config.encoder_path:
        fix = '_SNR_' + str(config.SNRs - 3)
        type = '_CBRs_'
    else:
        raise ValueError

    data_address = name + '_' + config.channel_type + type + 'JSCC' + fix
    snr, msssim = acquire_data(config, data_address, data_name=data_name)
    text = 'JSCC'
    plt.plot(snr, msssim, label=text, marker='^', markersize=10)

    data_address = name + '_' + config.channel_type + type + 'JSCC+CDDM' + fix
    snr, msssim = acquire_data(config, data_address, data_name=data_name)
    text = 'JSCC+CDDM'
    plt.plot(snr, msssim, label=text, marker='*', markersize=10)



    if config.channel_type == 'rayleigh':
        data_address = name + '_' + config.channel_type + type + 'JSCC' + '_h_sigma_' + str(
            config.h_sigma[0]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'JSCC' + '_h_sigma_' + str(config.h_sigma[0])
        plt.plot(snr, msssim, label=text, marker='o', markersize=10)

        data_address = name + '_' + config.channel_type + type + 'JSCC' + '_h_sigma_' + str(
            config.h_sigma[1]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'JSCC' + '_h_sigma_' + str(config.h_sigma[1])
        plt.plot(snr, msssim, label=text, marker='1', markersize=10)

        data_address = name + '_' + config.channel_type + type + 'JSCC+CDDM' + '_h_sigma_' + str(
            config.h_sigma[0]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'JSCC+CDDM' + '_h_sigma_' + str(config.h_sigma[0])
        plt.plot(snr, msssim, label=text, marker='8', markersize=10)

        data_address = name + '_' + config.channel_type + type + 'JSCC+CDDM' + '_h_sigma_' + str(
            config.h_sigma[1]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'JSCC+CDDM' + '_h_sigma_' + str(config.h_sigma[1])
        plt.plot(snr, msssim, label=text, marker='+', markersize=10)

    plt.xlabel("SNR (dB)", size=20)
    plt.ylabel(name + " (dB)", size=20)
    plt.title("Rayleigh fading channel", size=20)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(fontsize=20)

    plt.draw()
    plt.savefig('./data/{}_{}_{}_{}'.format(config.dataset, name, config.channel_type, type))


def plot_MSE(config):
    data_name = 2
    plt.figure(figsize=(15, 8))
    if config.loss_function == "MSE":
        name = 'PSNR'
    elif config.loss_function == "MSSSIM":
        name = "MSSSIM"
    else:
        raise ValueError

    if config.dataset == "CIFAR10":
        CBR = config.C / (12 * 8)
    elif config.dataset == "DIV2K":
        CBR = config.C / (96 * 12)
    elif config.dataset == "CelebA":
        CBR = config.C / (24 * 16)
    else:
        raise ValueError

    if 'SNRs' in config.encoder_path:
        fix = '_CBR_' + str(CBR)
        type = '_SNRs_'
        plt.xlabel("SNR (dB)", size=20)
    elif 'CBRs' in config.encoder_path:
        fix = '_SNR_' + str(config.SNRs - 3)
        type = '_CBRs_'
        plt.xlabel("CBR", size=20)
    else:
        raise ValueError

    data_address = 'MSE' + name + '_' + config.channel_type + type + 'JSCC' + fix
    print(data_address)
    snr, msssim = acquire_data(config, data_address, data_name=data_name)

    text = 'without CDDM'
    plt.plot(snr, msssim, label=text, marker='^', markersize=10)

    data_address = 'MSE' + name + '_' + config.channel_type + type + 'JSCC+CDDM' + fix
    snr, msssim = acquire_data(config, data_address, data_name=data_name)
    text = 'with CDDM'
    plt.plot(snr, msssim, label=text, marker='*', markersize=10)

    if config.channel_type == 'rayleigh':
        data_address = 'MSE' + name + '_' + config.channel_type + type + 'JSCC' + '_h_sigma_' + str(
            config.h_sigma[0]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'without CDDM' + '_h_sigma_' + str(config.h_sigma[0])
        plt.plot(snr, msssim, label=text, marker='.', markersize=10)

        data_address = 'MSE' + name + '_' + config.channel_type + type + 'JSCC' + '_h_sigma_' + str(
            config.h_sigma[1]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'without CDDM' + '_h_sigma_' + str(config.h_sigma[1])
        plt.plot(snr, msssim, label=text, marker='o', markersize=10)

        data_address = 'MSE' + name + '_' + config.channel_type + type + 'JSCC+CDDM' + '_h_sigma_' + str(
            config.h_sigma[0]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'with CDDM' + '_h_sigma_' + str(config.h_sigma[0])
        plt.plot(snr, msssim, label=text, marker='1', markersize=10)

        data_address = 'MSE' + name + '_' + config.channel_type + type + 'JSCC+CDDM' + '_h_sigma_' + str(
            config.h_sigma[1]) + fix
        snr, msssim = acquire_data(config, data_address, data_name=data_name)
        text = 'with CDDM' + '_h_sigma_' + str(config.h_sigma[1])
        plt.plot(snr, msssim, label=text, marker='8', markersize=10)


    elif config.channel_type == 'awgn':
        pass
    else:
        raise ValueError
        # plt.show()


    plt.ylabel(name+' MSE' + " (dB)", size=20)
    plt.title("Rayleigh fading channel", size=20)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(fontsize=20)
    plt.draw()
    plt.savefig('./data/MSE_{}_{}_{}_{}'.format(config.dataset, name, config.channel_type, type))


if __name__ == '__main__':
    from latent_DDIM_main import experiment, config, CHDDIM_config

    for index, dataset in enumerate(experiment.dataset):
        for loss in experiment.loss_function:
            for channel_type in experiment.channel_type:
                SNRs = experiment.train_snr + experiment.SNRs
                # print(SNRs)
                for SNR in SNRs:
                    basepath = r'E:\code\DDPM\semdif_revise\checkpoints'
                    encoder_path = basepath + r'\JSCC\{}\{}\SNRs\encoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                         SNR,
                                                                                                         channel_type,
                                                                                                         experiment.C_confirm[
                                                                                                             index])
                    decoder_path = basepath + r'\JSCC\{}\{}\SNRs\decoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                         SNR,
                                                                                                         channel_type,
                                                                                                         experiment.C_confirm[
                                                                                                             index])
                    re_decoder_path = basepath + r'\JSCC\{}\{}\SNRs\redecoder_snr{}_channel_{}_C{}.pt'.format(dataset,
                                                                                                              loss,
                                                                                                              SNR - 3,
                                                                                                              channel_type,
                                                                                                              experiment.C_confirm[
                                                                                                                  index])
                    CDDM_path = basepath + r'\CDDM\{}\{}\SNRs\CDDM_snr{}_channel_{}_C{}.pt'.format(dataset, loss, SNR,
                                                                                                   channel_type,
                                                                                                   experiment.C_confirm[
                                                                                                       index])
                    # dbcol_name=loss+"_"+channel_type+"_"+"JSCC"
                    JSCC_config = config(loss=loss, channel_type=channel_type, dataset=dataset, SNRs=SNR,
                                         C=experiment.C_confirm[index], encoder_path=encoder_path,
                                         decoder_path=decoder_path, re_decoder_path=re_decoder_path)
                    CDDM_config = CHDDIM_config(C=experiment.C_confirm[index], path=CDDM_path)
                plot_matric(JSCC_config)
                break
            break
        break
