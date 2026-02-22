import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor


def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a: Tensor, t: Tensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start: Tensor, t: Tensor, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start: Tensor, t: Tensor):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start: Tensor, x_t: Tensor, t: Tensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t: Tensor, t: Tensor, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t: Tensor, t: Tensor, clip_denoised=True):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def sample(self, model: nn.Module, image_size, batch_size=8, channels=3):
        # denoise: reverse diffusion
        shape = (batch_size, channels, image_size, image_size)
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)  # x_T ~ N(0, 1)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
            imgs.append(img)
        return imgs

    def train_losses(self, model, x_start: Tensor, t: Tensor):
        # compute train losses
        noise = torch.randn_like(x_start)  # random noise ~ N(0, 1)
        x_noisy = self.q_sample(x_start, t, noise=noise)  # x_t ~ q(x_t | x_0)
        predicted_noise = model(x_noisy, t)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss


class DDIM(GaussianDiffusion):
    """
    Denoising Diffusion Implicit Models (DDIM) sampler.
    Inherits from GaussianDiffusion and adds a DDIM sampling method.
    """
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        super().__init__(timesteps, beta_schedule)

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, image_size, batch_size=8, channels=3,
                    n_steps=50, eta=0.0, clip_denoised=True):
        """
        Sample using DDIM with a reduced number of steps.

        Args:
            model: noise prediction model (usually a UNet)
            image_size: spatial size of images (height = width)
            batch_size: number of images to sample
            channels: number of image channels
            n_steps: number of sampling steps (must be <= timesteps)
            eta: stochasticity parameter (0 -> deterministic, 1 -> DDPM-like)
            clip_denoised: whether to clip predicted x0 to [-1, 1]

        Returns:
            List of images at each DDIM step (including the final result).
        """
        device = next(model.parameters()).device
        shape = (batch_size, channels, image_size, image_size)

        # Create a sequence of timesteps from T to 0, spaced uniformly
        # We use n_steps points, including the first (T) and last (0)
        step_indices = torch.linspace(0, self.timesteps - 1, n_steps, dtype=torch.long, device=device)
        # Reverse so we go from T down to 0
        timesteps = torch.flip(step_indices, dims=[0])

        # Start from pure noise
        img = torch.randn(shape, device=device)   # x_T
        imgs = [img.cpu()]   # optionally store intermediate results

        # Iterate over the sequence, except the last step (t=0) which gives the final image
        for i in tqdm(range(len(timesteps) - 1), desc='DDIM sampling', total=len(timesteps)-1):
            t = timesteps[i]                # current step
            t_next = timesteps[i + 1]       # next step (smaller)

            # Prepare tensors of shape (batch_size,) for index extraction
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_next_batch = torch.full((batch_size,), t_next, device=device, dtype=torch.long)

            # 1. Predict noise using the model
            pred_noise = model(img, t_batch)

            # 2. Extract cumulative alphas for current and next step
            alpha_t = self._extract(self.alphas_cumprod, t_batch, img.shape)      # ᾱ_t
            alpha_next = self._extract(self.alphas_cumprod, t_next_batch, img.shape)  # ᾱ_s

            # 3. Compute sigma (stochastic component)
            # sigma = η * √((1-ᾱ_s)/(1-ᾱ_t)) * √(1-ᾱ_t/ᾱ_s)
            # This formulation ensures the variance matches DDPM when η=1.
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_next)
            # Clamp for numerical safety
            sigma = torch.clamp(sigma, min=0.0)

            # 4. Predict x0 from current x_t and predicted noise
            pred_x0 = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1., 1.)

            # 5. Compute direction pointing to x_t (the "predicted" part)
            # Direction coefficient: √(1-ᾱ_s - σ²)
            dir_coeff = torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0.0))
            dir_xt = dir_coeff * pred_noise

            # 6. Generate random noise if eta > 0, otherwise zero
            noise = torch.randn_like(img) if eta > 0 else 0.0

            # 7. Update to x_s (x_{t_next})
            img = torch.sqrt(alpha_next) * pred_x0 + dir_xt + sigma * noise

            imgs.append(img.cpu())

        return imgs
