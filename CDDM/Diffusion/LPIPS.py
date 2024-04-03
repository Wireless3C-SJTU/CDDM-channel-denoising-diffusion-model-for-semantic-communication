#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
@file:file
@time: time
"""
import lpips


class util_of_lpips():
    def __init__(self, net, use_gpu=True):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        # img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        # img1 = lpips.im2tensor(lpips.load_image(img2_path))
        #
        # if self.use_gpu:
        #     img0 = img0.cuda()
        #     img1 = img1.cuda()
        img0=img1_path
        img1=img2_path
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01

if __name__=='__main__':
    LPIPS=util_of_lpips('vgg')
    path1='./fake_image/AE_0.25_1.png'
    path2='./real_image/Original_1.png'
    sorce=LPIPS.calc_lpips(path1,path2)
    print(sorce)