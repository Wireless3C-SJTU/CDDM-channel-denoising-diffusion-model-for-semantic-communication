import dataplot
from Diffusion.Train import *

from train import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


class experiment():
    test="CDDM"
    loss_function = ["MSE"]
    channel_type = ["awgn"]
    dataset = ["DIV2K"]
    SNRs = [10]
    train_snr = [13]
    C_confirm = [36]
    C_CIFAR = [24,16, 12, 8]
    C_DIV2K = [36,24, 12]
    #C_CelebA = [48, 24, 16, 8]
    C=[C_DIV2K]
    CBR_snr=15
    large_snr=3
    noise_schedule=[1]
    Tmax=[10]



class config():
    def __init__(self, loss, channel_type, dataset, SNRs, C, encoder_path, decoder_path, re_decoder_path):
        self.loss_function = loss
        self.channel_type = channel_type
        self.database_address = "mongodb://localhost:27017"
        self.dataset = dataset
        self.SNRs = SNRs
        self.C = C
        self.seed = 1024
        self.CUDA = True
        self.device = torch.device("cuda:0")
        # self.device_ids = [0]  # 在服务器上修改
        # training details
        self.learning_rate = 0.0001
        self.h_sigma = [0.1, 0.05]
        self.all_SNRs = [20, 15, 10, 5]
        self.n_d_train=4
        if self.dataset == "CIFAR10":
            self.test_batch=100
            self.epoch = 200
            self.retrain_epoch=4
            self.retrain_save_model_freq=4
            self.save_model_freq = 200
            self.batch_size = 180*4  # 根据服务器修改
            self.CDDM_batch=100
            self.image_dims = (3, 32, 32)
            self.train_data_dir = r"/home/wutong/dataset/CIFAR10"
            self.test_data_dir = r"/home/wutong/dataset/CIFAR10"
            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8],
                window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=False,
            )
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
                window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=False,
            )

        elif self.dataset == "DIV2K":
            self.test_batch=1
            self.epoch = 600  # 有待今晚确定
            self.retrain_epoch=20
            self.retrain_save_model_freq=20
            self.save_model_freq = 600
            self.batch_size = 4  # 根据服务器修改
            self.CDDM_batch=16
            
            self.image_dims = (3, 256, 256)
            self.train_data_dir = r"/mnt/wutong/datasets/DIV2K/DIV2K_train_HR"
            self.test_data_dir = r"/mnt/wutong/datasets/DIV2K/DIV2K_valid_HR"
            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )
        elif self.dataset == "CelebA":
            self.epoch = 1
            self.save_model_freq = 1
            self.batch_size = 50  # 根据服务器修改

            self.image_dims = (3, 128, 128)
            self.train_data_dir = r"D:\dateset\CelebA\Img\trainset"
            self.test_data_dir = r"D:\dateset\CelebA\Img\validset"
            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256], depths=[2, 2, 6], num_heads=[4, 6, 8],
                window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[256, 192, 128], depths=[6, 2, 2], num_heads=[8, 6, 4],
                window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )

        # batch_size = 100
        # downsample = 4
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.re_decoder_path = re_decoder_path


class CHDDIM_config():
    device_ids = [0]
    epoch = 400
    save_model_freq = 400
    T = 1000
    channel_mult = [1, 2, 2]
    attn = [1]
    num_res_blocks = 2
    dropout = 0.1
    lr = 1e-4
    multiplier = 2.
    snr_max = 1e-4
    snr_min = 0.02
    grad_clip = 1
    # equ = None
    device = "cuda"
    re_weight=True


    def __init__(self, C, path,large_snr,noise_schedule,t_max):
        self.C = C
        # img_size = 8
        self.t_max=t_max
        self.noise_schedule=noise_schedule
        self.channel = int(16. * C)  # 这里需要改
        # self.train_load_weight=CDDM_path
        self.save_path = path
        self.large_snr=large_snr


if __name__ == '__main__':
    for index, dataset in enumerate(experiment.dataset):
        for loss in experiment.loss_function:
            for channel_type in experiment.channel_type:
                SNRs = experiment.train_snr+experiment.SNRs
                # print(SNRs)
                for SNR in SNRs:
                    for noise_schedule in experiment.noise_schedule:
                        for t_max in experiment.Tmax:
                            basepath = r'/mnt/wutong/CDDMcheckpoints/checkpoints' # r'/home/wutong/semdif_revise/checkpoints'
                            if noise_schedule==1:

                                encoder_path = basepath + r'/JSCC/{}/{}/SNRs/encoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[
                                                                                                                        index])
                                decoder_path = basepath + r'/JSCC/{}/{}/SNRs/decoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[
                                                                                                                        index])
                                if experiment.test=="CDDM":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                elif experiment.test=="small":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_small.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_small.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                elif experiment.test=="lessT":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_lessT.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_lessT.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                elif experiment.test=="re-weight":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_noweight.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_noreweight.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                elif experiment.test=="Tmax":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_tmax{}.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index],t_max)
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                elif experiment.test=="DnCNN":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_DnCNN.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_DnCNN.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                elif experiment.test=="GAN":
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_GAN.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[
                                                                                                                                index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_GAN.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[
                                                                                                                    index])
                                else:
                                    raise ValueError
                                                                                                                
                            else:
                                
                                encoder_path = basepath + r'/JSCC/{}/{}/SNRs/encoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[
                                                                                                                        index])
                                decoder_path = basepath + r'/JSCC/{}/{}/SNRs/decoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[
                                                                                                                        index])
                                re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_ns{}.pt'.format(dataset,
                                                                                                                        loss,
                                                                                                                        SNR - 3,
                                                                                                                        channel_type,
                                                                                                                        experiment.C_confirm[
                                                                                                                            index],noise_schedule)
                                CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_ns{}.pt'.format(dataset, loss, SNR,
                                                                                                            channel_type,
                                                                                                            experiment.C_confirm[
                                                                                                                index],noise_schedule)
                            # dbcol_name=loss+"_"+channel_type+"_"+"JSCC"
                            JSCC_config = config(loss=loss, channel_type=channel_type, dataset=dataset, SNRs=SNR,
                                                C=experiment.C_confirm[index], encoder_path=encoder_path,
                                                decoder_path=decoder_path, re_decoder_path=re_decoder_path)
                            CDDM_config = CHDDIM_config(C=experiment.C_confirm[index], path=CDDM_path,large_snr=experiment.large_snr,noise_schedule=noise_schedule,t_max=t_max)
                            seed_torch()
                            # train_JSCC_seqeratly(JSCC_config)
                            eval_only_JSCC(JSCC_config)
                            # if SNR == max(experiment.SNRs):
                            #     eval_JSCC_SNRs(JSCC_config)
                            # if channel_type == 'rayleigh':
                            #     #eval_only_JSCC_delte_h(JSCC_config)
                            #     pass
                            if SNR in experiment.train_snr:
                                print(experiment.test)
                                if experiment.test=="DnCNN":
                                    seed_torch()
                                    #train_DnCNN(JSCC_config, CDDM_config)
                                    seed_torch()
                                    #train_JSCC_with_DnCNN(JSCC_config, CDDM_config)
                                    seed_torch()
                                    eval_JSCC_with_DnCNN(JSCC_config, CDDM_config)
                                elif experiment.test=="GAN":
                                    seed_torch()
                                    train_GAN(JSCC_config,CDDM_config)
                                    seed_torch()
                                    train_JSCC_with_GAN(JSCC_config,CDDM_config)
                                    seed_torch()
                                    eval_JSCC_with_GAN(JSCC_config,CDDM_config)
                                else:
                                
                                    seed_torch()
                                    #train_CHDDIM(JSCC_config, CDDM_config)
                                    seed_torch()
                                    #train_JSCC_with_CDDM(JSCC_config, CDDM_config)
                                    seed_torch()
                                    eval_JSCC_with_CDDM(JSCC_config, CDDM_config)

                                    

    