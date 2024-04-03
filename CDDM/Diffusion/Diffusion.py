import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from Diffusion.Autoencoder import noise_encoder
def sigmoid_schedule(T,start=-3,end=3,tau=1.0,clip_min=1e-9):
    t=torch.linspace(0, T, T).double()
    start=torch.tensor(start)
    end=torch.tensor(end)
    v_start=torch.sigmoid(start/tau)
    v_end=torch.sigmoid(end/tau)
    output=torch.sigmoid((t/T*(end-start)+start)/tau)
    output=(v_end-output)/(v_end-v_start)
    return torch.clip(output,clip_min,1.)

def cosine_schedule(T,start=-3,end=3,tau=1.0,clip_min=1e-9):
    t=torch.linspace(0, T, T).double()
    start=torch.tensor(start)
    end=torch.tensor(end)
    v_start=torch.cos(start*math.pi/2)**(2*tau)
    v_end=torch.cos(end*math.pi/2)**(2*tau)
    output=torch.cos((t/T*(end-start)+start)*math.pi/2)**(2*tau)
    output=(v_end-output)/(v_end-v_start)
    return torch.clip(output,clip_min,1.)

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class ChannelDiffusionTrainer(nn.Module):
    def __init__(self, model,noise_schedule, re_weight, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        self.re_weight=re_weight
        if noise_schedule==1:
            self.register_buffer(
                'betas', torch.linspace(beta_1, beta_T, T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)

            # calculations for diffusion q(x_t | x_{t-1}) and others

        elif noise_schedule==2:
        
            alphas_bar=cosine_schedule(T,0,1,1)
        elif noise_schedule==3:
            alphas_bar=cosine_schedule(T,0.2,1,3)
        elif noise_schedule==4:
            alphas_bar=cosine_schedule(T,0.2,1,1)
        elif noise_schedule==5:
            alphas_bar=sigmoid_schedule(T,0,3,0.3)
        elif noise_schedule==6:
            alphas_bar=sigmoid_schedule(T,0,3,0.7)
        elif noise_schedule==7:
            alphas_bar=sigmoid_schedule(T,-3,3,10)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, h, snr,channel_type):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        print(t)
        #t = torch.randint(93, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        sigma_square = 1.0 / (10 ** (snr / 10))

        # if equ=="ZF":
        #     h = torch.abs(h)
        #     h = torch.cat((h, h), dim=2)
        #     x_t = (
        #             extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
        #             extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise / h)
        if channel_type=="rayleigh":
            h = torch.abs(h)
            h_signal = torch.cat((h ** 2 / (h ** 2 + sigma_square), h ** 2 / (h ** 2 + sigma_square)), dim=2)
            h_noise = torch.cat((h / (h ** 2 + sigma_square), h / (h ** 2 + sigma_square)), dim=2)
            h = torch.cat((h, h), dim=2)

            x_t = (
                    extract(self.sqrt_alphas_bar, t, x_0.shape) * (h_signal*x_0) +
                    extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise *h_noise)
        elif channel_type=='awgn':
            x_t = (
                    extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        else:
            raise ValueError

        out = self.model(x_t, t, h)

        # out = out.reshape(B, H, W)
        # print(x_t.shape)
        if self.re_weight==True:
            loss = nn.MSELoss()(out, noise)
        elif self.re_weight==False:
            loss= nn.MSELoss()(h_noise*out, h_noise*noise)
        else:
            raise ValueError
        return loss


class ChannelDiffusionSampler(nn.Module):
    def __init__(self, model,noise_schedule,t_max, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        # self.AE.train()
        self.t_max=t_max
        if noise_schedule==1:
            self.register_buffer(
                'betas', torch.linspace(beta_1, beta_T, T).double())
            alphas = 1. - self.betas
            alphas_bar = torch.cumprod(alphas, dim=0)

            # calculations for diffusion q(x_t | x_{t-1}) and others

        elif noise_schedule==2:
            alphas_bar=cosine_schedule(T,0,1,1)
        elif noise_schedule==3:
            alphas_bar=cosine_schedule(T,0.2,1,3)
        elif noise_schedule==4:
            alphas_bar=cosine_schedule(T,0.2,1,1)
        elif noise_schedule==5:
            alphas_bar=sigmoid_schedule(T,0,3,0.3)
        elif noise_schedule==6:
            alphas_bar=sigmoid_schedule(T,0,3,0.7)
        elif noise_schedule==7:
            alphas_bar=sigmoid_schedule(T,-3,3,10)

        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        # self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        # self.register_buffer('coeff3', self.coeff1 * (1. - alphas))

        # self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('snr', -10 * torch.log10((1 - alphas_bar) / alphas_bar))

    # def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
    #     assert x_t.shape == eps.shape
    #     return (
    #             extract(self.coeff1, t, x_t.shape) * x_t -
    #             extract(self.coeff2, t, x_t.shape) * eps
    #     )

    # def p_mean_variance(self, x_t, t):
    #     # below: only log_variance is used in the KL computations
    #     var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
    #     var = extract(var, t, x_t.shape)

    #     eps = self.model(x_t, t)

    #     xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

    #     return xt_prev_mean, var

    def match_snr_t(self, snr):
        out = torch.argmin(torch.abs(self.snr - snr))
        return out

    def forward(self, x_hat, snr, snr_train, h, channel_type):
        """
        Algorithm 2.
        """
        snr=max(snr,self.t_max)
        #10 for CIFAR 15 for DIV2K
        t = self.match_snr_t(snr) + 1

        
        #print(t)
        sigma_square_fix = 1.0 / (10 ** (snr_train / 10))
        # print(sigma_square)
        if channel_type == 'rayleigh':
            x_t = x_hat * (torch.conj(h)) / (torch.abs(h) ** 2 + sigma_square_fix)
        elif channel_type=='awgn':
            x_t = x_hat
        else:
            raise ValueError
        x_t = torch.cat((torch.real(x_t), torch.imag(x_t)), dim=2)

        h = torch.abs(h)
        h = torch.cat((h, h), dim=2)

        for time_step in reversed(range(t)):
            # print(time_step)
            t = x_t.new_ones([x_hat.shape[0], ], dtype=torch.long) * time_step
            eps = self.model(x_t, t, h)

            if channel_type == 'rayleigh':
                eps = eps * (h / (h ** 2 + sigma_square_fix))
            elif channel_type=='awgn':
                eps = eps
            else:
                raise ValueError

            if time_step > 0:

                x_t = extract(self.sqrt_alphas_bar, t - 1, x_t.shape) / extract(self.sqrt_alphas_bar, t, x_t.shape) * (
                        x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps) + extract(
                    self.sqrt_one_minus_alphas_bar, t - 1, x_t.shape) * eps
            else:
                x_t = (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps) / extract(
                    self.sqrt_alphas_bar, t, x_t.shape)

            x_t = x_t.detach()
            x_temp = x_t
            x_t = x_temp.detach()

            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t

        return x_0

