import pymongo
import torch.optim as optim

from Autoencoder.data.datasets import get_loader
from Autoencoder.net.channel import Channel
from Autoencoder.net.network import JSCC_encoder, JSCC_decoder
from Autoencoder.utils import *
from torchvision.utils import save_image
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import torch.nn as nn
import argparse
from Autoencoder.loss.distortion import *
import time
import sys
from tqdm import tqdm

# print(torch.cuda.is_available())
parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true', default=True,
                    help='training or testing')

args = parser.parse_args()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# class config():
#     channel_type = "awgn"
#     dataset = "CIFAR10"
#     seed = 1024
#     pass_channel = True
#     CUDA = True
#     device = torch.device("cuda:0")
#     device_ids = [0]
#     norm = False
#     if_sample = False
#     # logger
#     print_step = 39
#     plot_step = 10000
#     filename = datetime.now().__str__()[:-16]
#     workdir = './Autoencoder/history/{}'.format(filename)
#     log = workdir + '/Log_{}.log'.format(filename)
#     samples = workdir + '/samples'
#     models = 'E:\code\DDPM\SemDiffusion\Autoencoder\history'
#     logger = None
#     equ = "MMSE"
#     # training details
#     normalize = False
#     learning_rate = 0.0001
#     epoch = 20
#
#     save_model_freq = 20
#     if dataset == "CIFAR10":
#         image_dims = (3, 32, 32)
#         train_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
#         test_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
#         encoder_kwargs = dict(
#             img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
#             embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8],
#             window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#             norm_layer=nn.LayerNorm, patch_norm=False,
#         )
#         decoder_kwargs = dict(
#             img_size=(image_dims[1], image_dims[2]),
#             embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
#             window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#             norm_layer=nn.LayerNorm, patch_norm=False,
#         )
#
#     elif dataset == "DIV2K":
#         image_dims = (3, 256, 256)
#         train_data_dir = r"D:\dateset\DIV2K\DIV2K_train_HR"
#         test_data_dir = r"D:\dateset\DIV2K\DIV2K_valid_HR"
#         encoder_kwargs = dict(
#             img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
#             embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
#             window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#             norm_layer=nn.LayerNorm, patch_norm=True,
#         )
#         decoder_kwargs = dict(
#             img_size=(image_dims[1], image_dims[2]),
#             embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
#             window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#             norm_layer=nn.LayerNorm, patch_norm=True,
#         )
#     elif dataset == "CelebA":
#         image_dims = (3, 128, 128)
#         train_data_dir = r"D:\dateset\CelebA\Img\trainset"
#         test_data_dir = r"D:\dateset\CelebA\Img\validset"
#     batch_size = 1
#     # batch_size = 100
#     downsample = 4
#     C = [12]
#
#     encoder_path = r"/home/temp/JSCCmodel/SNR{}{}_encoder200CBR0.125.model".format(12, equ)
#     decoder_path = r"/home/temp/JSCCmodel/SNR{}{}_decoder200CBR0.125.model".format(12, equ)
#     # encoder_path = r"E:\code\DDPM\SemDiffusion\Autoencoder\history\2023-04-08\models\SNR10{}_encoder200CBR0.125.model".format(equ)
#     # decoder_path = r"E:\code\DDPM\SemDiffusion\Autoencoder\history\2023-04-08\models\SNR10{}_decoder200CBR0.125.model".format(equ)


def train_JSCC_seqeratly(config):
    encoder = JSCC_encoder(config, config.C).cuda()
    decoder = JSCC_decoder(config, config.C).cuda()
    channel = Channel(config)

    # encoder = torch.nn.DataParallel(encoder, device_ids=config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=config.device_ids)

    # encoder_path = r"E:\code\DDPM\semdif_revise\DIV2k\DIV2KSNR10None_encoder200CBR0.0625.model"
    # decoder_path = r"E:\code\DDPM\semdif_revise\DIV2k\DIV2KSNR10None_decoder200CBR0.0625.model"
    # encoder.load_state_dict(torch.load(encoder_path))
    # decoder.load_state_dict(torch.load(decoder_path))

    # encoder = encoder.cuda(device=config.device_ids[0])
    # decoder = decoder.cuda(device=config.device_ids[0])

    train_loader, _ = get_loader(config)
    cur_lr = config.learning_rate
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=cur_lr)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=cur_lr)
    encoder.train()
    decoder.train()
    #test_mem_and_comp(config,encoder,decoder)
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
        kl_weight = 5e-5
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
        #kl_weight = 1e-6
    seed_torch()
    for e in range(config.epoch):
        with tqdm(train_loader, dynamic_ncols=True) as tqdmTrainData:
            for input, label in tqdmTrainData:
                start_time = time.time()
                input = input.cuda()
                # print(input.shape)
                feature, _ = encoder(input)
                y = feature
                #print(feature.shape)
                SNR = config.SNRs
                CBR = feature.numel() / 2 / input.numel()
                # print(y.shape)
                noisy_y, pwr, h = channel.forward(y, SNR)
                if config.channel_type == "rayleigh":
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
                elif config.channel_type == "awgn":
                    pass
                else:
                    raise ValueError

                noisy_y = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2) * torch.sqrt(pwr)
                # noisy_feature=feature
                recon_image = decoder(noisy_y)

                if config.loss_function == "MSE":
                    mse = nn.MSELoss()(input * 255., recon_image.clamp(0., 1.) * 255)
                    rec_loss = F.mse_loss(recon_image.clamp(0., 1.), input, reduction="sum") / input.shape[0]
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10)).item()
                    matric = psnr
                elif config.loss_function == "MSSSIM":
                    rec_loss = CalcuSSIM(input, recon_image.clamp(0., 1.)).mean() * input.numel() / input.shape[0]
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    matric = msssim
                else:
                    raise ValueError
                # kl_loss = kl
                # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss = rec_loss #+ kl_weight * kl_loss
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward()
                optimizer_decoder.step()
                optimizer_encoder.step()

                # print(type(psnr))
                # DLweichat.weichat_send("psnr")

                tqdmTrainData.set_postfix(ordered_dict={
                    "epoch:": e,
                    "state": 'train' + config.loss_function,
                    "dataset": config.dataset,
                    "channel": config.channel_type,
                    "SNR:": SNR,
                    "CBR:": CBR,
                    "matric": matric,
                })

        if (e + 1) % config.save_model_freq == 0:
            save_model(encoder, save_path=config.encoder_path)
            save_model(decoder, save_path=config.decoder_path)
            # test()
            # print(1)


def eval_only_JSCC(config):
    encoder = JSCC_encoder(config, config.C).cuda()
    decoder = JSCC_decoder(config, config.C).cuda()
    channel = Channel(config)

    # encoder = torch.nn.DataParallel(encoder, device_ids=config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=config.device_ids)

    # encoder = encoder.cuda(device=config.device_ids[0])
    # decoder = decoder.cuda(device=config.device_ids[0])
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    _, test_loader = get_loader(config)

    encoder.eval()
    decoder.eval()
    #test_mem_and_comp(config,encoder,decoder)
    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

    matric_aver = 0
    seed_torch()
    with tqdm(test_loader, dynamic_ncols=True) as tqdmtestData:
        for i, (input, label) in enumerate(tqdmtestData):
            start_time = time.time()
            input = input.cuda()
            # print(input.shape)
            feature, _ = encoder(input)
            y = feature

            SNR = config.SNRs
            CBR = feature.numel() / 2 / input.numel()

            noisy_y, pwr, h = channel.forward(y, SNR)
            if config.channel_type == "rayleigh":
                sigma_square = 1.0 / (10 ** (SNR / 10))
                noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
            elif config.channel_type == "awgn":
                pass
            else:
                raise ValueError
            noisy_y = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2) * torch.sqrt(pwr)

            recon_image = decoder(noisy_y)
            if config.loss_function == "MSE":
                mse = nn.MSELoss()(input * 255., recon_image.clamp(0., 1.) * 255)
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10)).item()
                matric = psnr
                save_image(recon_image,"/home/wutong/semdif_revise/DIV2K_JSCC_rayleigh_PSNR_10dB/{}.png".format(i))
            elif config.loss_function == 'MSSSIM':
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                matric = msssim
                save_image(recon_image,"/home/wutong/semdif_revise/DIV2K_JSCC_rayleigh_MSSSIM_10dB/{}.png".format(i))
            else:
                raise ValueError
            # print(type(psnr))
            # DLweichat.weichat_send("psnr")

            tqdmtestData.set_postfix(ordered_dict={
                "dataset": config.dataset,
                "state": "eval" + config.loss_function,
                "channel": config.channel_type,
                "CBR:": CBR,
                "SNR": SNR,
                "matric": matric,
            })
            matric_aver += matric
        matric_aver = matric_aver / (i + 1)
        if config.loss_function == "MSE":
            name = 'PSNR'
        elif config.loss_function == "MSSSIM":
            name = "MSSSIM"
        else:
            raise ValueError
        myclient = pymongo.MongoClient(config.database_address)
        mydb = myclient[config.dataset]
        if 'SNRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_SNRs_' + 'JSCC' + '_CBR_' + str(CBR)]
            mydic = {'SNR': SNR, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)
        elif 'CBRs' in config.encoder_path:
            mycol = mydb[name + '_' + config.channel_type + '_CBRs_' + 'JSCC' + '_SNR_' + str(SNR)]
            mydic = {'CBR': CBR, name: matric_aver}
            mycol.insert_one(mydic)
            print('writing successfully', mydic)
        else:
            raise ValueError


def eval_only_JSCC_delte_h(config):
    encoder = JSCC_encoder(config, config.C).cuda()
    decoder = JSCC_decoder(config, config.C).cuda()
    channel = Channel(config)

    # encoder = torch.nn.DataParallel(encoder, device_ids=config.device_ids)
    # decoder = torch.nn.DataParallel(decoder, device_ids=config.device_ids)

    # encoder = encoder.cuda(device=config.device_ids[0])
    # decoder = decoder.cuda(device=config.device_ids[0])
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    _, test_loader = get_loader(config)

    encoder.eval()
    decoder.eval()

    if config.dataset == "CIFAR10":
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    for h_sigma in config.h_sigma:
        matric_aver = 0
        with tqdm(test_loader, dynamic_ncols=True) as tqdmtestData:
            for i, (input, label) in enumerate(tqdmtestData):
                start_time = time.time()
                input = input.cuda()
                # print(input.shape)
                feature, _ = encoder(input)
                y = feature

                SNR = config.SNRs
                CBR = feature.numel() / 2 / input.numel()

                noisy_y, pwr, h = channel.forward(y, SNR)

                delte_h = h_sigma * (
                        torch.normal(mean=0.0, std=1, size=np.shape(h)) + 1j * torch.normal(mean=0.0, std=1,
                                                                                            size=np.shape(
                                                                                                h))) / np.sqrt(2)
                h = h + delte_h.cuda()
                if config.channel_type == "rayleigh":
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
                elif config.channel_type == "awgn":
                    pass
                else:
                    raise ValueError
                noisy_y = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2) * torch.sqrt(pwr)

                recon_image = decoder(noisy_y)
                if config.loss_function == "MSE":
                    mse = nn.MSELoss()(input * 255., recon_image.clamp(0., 1.) * 255)
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10)).item()
                    matric = psnr

                elif config.loss_function == 'MSSSIM':
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    matric = msssim
                else:
                    raise ValueError
                # print(type(psnr))
                # DLweichat.weichat_send("psnr")

                tqdmtestData.set_postfix(ordered_dict={
                    "dataset": config.dataset,
                    "state": "eval_delte_h_" + config.loss_function,
                    "h sigma":h_sigma,
                    "channel": config.channel_type,
                    "CBR:": CBR,
                    "SNR": SNR,
                    "matric": matric,
                })
                matric_aver += matric
            matric_aver = matric_aver / (i + 1)
            if config.loss_function == "MSE":
                name = 'PSNR'
            elif config.loss_function == "MSSSIM":
                name = "MSSSIM"
            else:
                raise ValueError
            myclient = pymongo.MongoClient(config.database_address)
            mydb = myclient[config.dataset]
            if 'SNRs' in config.encoder_path:
                mycol = mydb[
                    name + '_' + config.channel_type + '_SNRs_' + 'JSCC' + '_h_sigma_' + str(h_sigma) + '_CBR_' + str(
                        CBR)]
                mydic = {'SNR': SNR, name: matric_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)
            elif 'CBRs' in config.encoder_path:
                mycol = mydb[
                    name + '_' + config.channel_type + '_CBRs_' + 'JSCC' + '_h_sigma_' + str(h_sigma) + '_SNR_' + str(
                        SNR)]
                mydic = {'CBR': CBR, name: matric_aver}
                mycol.insert_one(mydic)
                print('writing successfully', mydic)
            else:
                raise ValueError

        # if i==0:
        #     save_image(recon_image, os.path.join(
        #         "E:\code\DDPM\SemDiffusion", "VWITT_" + str(i) + '.png'), nrow=10)
        # break
            
def test_mem_and_comp(config,encoder,decoder):
    from thop import profile
    from thop import clever_format
    class net(torch.nn.Module):

        def __init__(self,encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self,input):

            SNR=20
            x=self.encoder(input)
            y=self.decoder(x)
            return y
    
    network=net(encoder,decoder).cuda()
    input=torch.randn(1,3,config.image_dims[1],config.image_dims[2]).cuda()
    #print(input.shape)
    macs,params=profile(network,inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    torch.cuda.empty_cache()
    del network
    torch.cuda.empty_cache()
    print(macs,params)
