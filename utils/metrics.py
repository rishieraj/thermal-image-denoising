import torch
from pytorch_msssim import SSIM, MS_SSIM

class MetricCalculator:
    """Handles all quality metrics calculation"""
    def __init__(self, device='cuda'):
        self.ssim = SSIM(data_range=1.0, channel=1).to(device)
        self.ms_ssim = MS_SSIM(data_range=1.0, channel=1).to(device)

    def calculate(self, preds, targets):
        return {
            'SSIM': self.ssim(preds, targets),
            'MS-SSIM': self.ms_ssim(preds, targets),
            'PSNR': 10 * torch.log10(1 / F.mse_loss(preds, targets))
        }