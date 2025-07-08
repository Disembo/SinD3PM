import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDPM:

    def __init__(self,
                 device,
                 n_steps: int = 1000,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02,
                 pred_x0: bool = True,
                 simple_var: bool = True):
        self.device = device
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.hstack([torch.tensor([1.0], device=device), self.alpha_bars[:-1]])
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_recip_alpha_bars = torch.sqrt(1.0 / self.alpha_bars)
        self.sqrt_recipm1_alpha_bars = torch.sqrt(1.0 / self.alpha_bars - 1)
        
        if simple_var:
            self.posterior_variance = self.betas.clone()
            self.posterior_variance[0] = 0
        else:
            self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)
        
        self.n_steps = n_steps
        self.pred_x0 = pred_x0
        
    def sample_forward(self, x: torch.Tensor, t: int | torch.Tensor, eps=None):
        if eps is None:
            eps = torch.randn_like(x)
        coef1 = self.sqrt_alpha_bars[t].reshape(-1, 1, 1, 1)
        coef2 = self.sqrt_one_minus_alpha_bars[t].reshape(-1, 1, 1, 1)
        res = coef1 * x + coef2 * eps
        return res
    
    def infer(self, n_sample: int, net: nn.Module, size: tuple[int, int], channels: int):
        img_shape = (n_sample, channels, *size)
        x = torch.randn(img_shape).to(self.device)
        with torch.no_grad():
            for t in tqdm(range(self.n_steps - 1, -1, -1), desc='Diffusion sampling'):
                x = self.sample_backward_step(x, t, net)
        return x

    def sample_backward_step(self, x_t, t, net):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device)
        out = net(x_t, t_tensor)
        var = self.posterior_variance[t]
        if self.pred_x0:
            x_0 = out
        else:
            x_0 = self.sqrt_recip_alpha_bars[t] * x_t - self.sqrt_recipm1_alpha_bars[t] * out
        mean = self.posterior_mean_coef1[t] * x_0 + self.posterior_mean_coef2[t] * x_t
        noise = torch.randn_like(x_t)
        noise *= torch.sqrt(var)
        x_prev = mean + noise
        return x_prev
    
    def calc_loss(self, net: nn.Module, x_0: torch.Tensor):
        x_0 = x_0.to(self.device)
        t = torch.randint(0, self.n_steps, (x_0.shape[0],)).to(self.device)
        eps = torch.randn_like(x_0)
        x_t = self.sample_forward(x_0, t, eps)
        out = net(x_t, t)
        if self.pred_x0:
            loss = F.mse_loss(out, x_0)
        else:
            loss = F.mse_loss(out, eps)
        return loss
