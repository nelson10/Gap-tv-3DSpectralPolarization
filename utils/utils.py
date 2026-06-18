import torch
import torch.nn as nn
import numpy as np
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage import img_as_ubyte
import sys

def clip(x):
    if not isinstance(x, np.ndarray):
        x = x.clone().cpu().numpy()
    return x.clip(0., 1.)

def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''
    if im1.ndim == 2:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
        # out = compare_ssim(im1, im2, data_range=255, gaussian_weights=True,
        #                                             use_sample_covariance=False, multichannel=False)                                           
    elif im1.ndim == 3:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
        # out = compare_ssim(im1, im2, data_range=255, gaussian_weights=True,
        #                                              use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out

def ssim(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    SSIM = ssim_index(img, img_clean)
    return SSIM

def psnr(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    PSNR = peak_signal_noise_ratio(img, img_clean, data_range=255)
    # PSNR = compare_psnr(img, img_clean, data_range=255)
    return PSNR

def calculate_sam(img1, img2):
    """
    Calcula el Spectral Angle Mapper (SAM) entre dos cubos [H, W, C].
    Implementación vectorizada para PyTorch.
    """
    # Valor pequeño para evitar divisiones por cero (similar al eps de tu MATLAB)
    eps = 1e-8
    
    # Producto punto por píxel a través de las bandas (dim=2)
    dot_product = torch.sum(img1 * img2, dim=2)
    
    # Normas de cada vector espectral por píxel
    norm1 = torch.norm(img1, dim=2)
    norm2 = torch.norm(img2, dim=2)
    
    # Cálculo del coseno y estabilidad numérica para acos
    cos_theta = dot_product / (norm1 * norm2 + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Ángulo en radianes
    ang = torch.acos(cos_theta)
    
    # Retorna el promedio de todos los píxeles del cubo
    return torch.mean(ang).item()