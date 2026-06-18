import numpy as np
import torch

def TV_denoising(y0, lamda, iteration=100):
    # z = torch.zeros(y0.shape - [1, 1, 1], device=device, dtype=torch.float32)
    device = y0.device
    w, h, b  = y0.shape
    zh = torch.zeros([w, h-1, b], device=device, dtype=torch.float32)
    zv = torch.zeros([w-1, h, b], device=device, dtype=torch.float32)
    alpha = 5
    for it in range(iteration):
        x0h = y0 - dht_3d(zh)
        x0v = y0 - dvt_3d(zv)
        x0 = (x0h + x0v) / 2
        zh = clip(zh + 1/alpha*dh(x0), lamda/2)
        zv = clip(zv + 1/alpha*dv(x0), lamda/2)
    return x0

def TV_denoising3d(y0, lamda, iteration=100):
    device = y0.device
    # z = torch.zeros(y0.shape - [1, 1, 1], device=device, dtype=torch.float32)
    w, h, b  = y0.shape
    zh = torch.zeros([w, h-1, b], device=device, dtype=torch.float32)
    zv = torch.zeros([w-1, h, b], device=device, dtype=torch.float32)
    zt = torch.zeros([w, h, b-1], device=device, dtype=torch.float32)
    alpha = 5
    for it in range(iteration):
        x0h = y0 - dht_3d(zh)
        x0v = y0 - dvt_3d(zv)
        x0t = y0 - dtt_3d(zt)
        x0 = (x0h + x0v + x0t) / 3
        zh = clip(zh + 1/alpha*dh(x0), lamda/2)
        zv = clip(zv + 1/alpha*dv(x0), lamda/2)
        zt = clip(zt + 1/alpha*dt(x0), lamda/2)
    return x0

def clip(x, thres):
    return torch.clamp(x, min=-thres, max=thres)

def dht_3d(x):
    return torch.cat([-x[:,0:1,:], x[:,:-1,:]-x[:,1:,:], x[:,-1:,:]], 1)

def dvt_3d(x):
    return torch.cat([-x[0:1,:,:], x[:-1,:,:]-x[1:,:,:], x[-1:,:,:]], 0)

def dtt_3d(x):
    return torch.cat([-x[:,:,0:1], x[:,:,:-1]-x[:,:,1:], x[:,:,-1:]], 2)

def dh(x):
    return x[:,1:,:]-x[:,:-1,:]

def dv(x):
    return x[1:,:,:] - x[:-1,:,:]

def dt(x):
    return x[:,:,1:] - x[:,:,:-1]

# --- 4D Gradient Operators (Finite Differences) ---
def dh_4d(x): return x[1:, :, :, :] - x[:-1, :, :, :] # Height gradient
def dw_4d(x): return x[:, 1:, :, :] - x[:, :-1, :, :] # Width gradient
def dp_4d(x): return x[:, :, 1:, :] - x[:, :, :-1, :] # Polarization gradient
def ds_4d(x): return x[:, :, :, 1:] - x[:, :, :, :-1] # Spectral gradient

# --- 4D Adjoint Operators (Negative Divergence) ---
def dht_4d(z):
    h, w, p, s = z.shape
    res = torch.zeros((h + 1, w, p, s), device=z.device, dtype=z.dtype)
    res[0, :, :, :] = -z[0, :, :, :]
    res[1:-1, :, :, :] = z[:-1, :, :, :] - z[1:, :, :, :]
    res[-1, :, :, :] = z[-1, :, :, :]
    return res

def dwt_4d(z):
    h, w, p, s = z.shape
    res = torch.zeros((h, w + 1, p, s), device=z.device, dtype=z.dtype)
    res[:, 0, :, :] = -z[:, 0, :, :]
    res[:, 1:-1, :, :] = z[:, :-1, :, :] - z[:, 1:, :, :]
    res[:, -1, :, :] = z[:, -1, :, :]
    return res

def dpt_4d(z):
    h, w, p, s = z.shape
    res = torch.zeros((h, w, p + 1, s), device=z.device, dtype=z.dtype)
    res[:, :, 0, :] = -z[:, :, 0, :]
    res[:, :, 1:-1, :] = z[:, :, :-1, :] - z[:, :, 1:, :]
    res[:, :, -1, :] = z[:, :, -1, :]
    return res

def dst_4d(z):
    h, w, p, s = z.shape
    res = torch.zeros((h, w, p, s + 1), device=z.device, dtype=z.dtype)
    res[:, :, :, 0] = -z[:, :, :, 0]
    res[:, :, :, 1:-1] = z[:, :, :, :-1] - z[:, :, :, 1:]
    res[:, :, :, -1] = z[:, :, :, -1]
    return res

# --- Main 4D TV Denoising Function ---
def TV_denoising4d(y0, lamda, iteration=100):
    """
    Total Variation Denoising for 4D hypercubes: [H, W, Polarization, Spectrum]
    
    Args:
        y0: Input noisy 4D tensor.
        lamda: Regularization parameter (smoothing strength).
        iteration: Number of dual iterations.
    """
    device = y0.device
    h, w, p, s = y0.shape
    
    # Initialize dual variables for each of the 4 dimensions
    zh = torch.zeros([h-1, w, p, s], device=device, dtype=torch.float32)
    zw = torch.zeros([h, w-1, p, s], device=device, dtype=torch.float32)
    zp = torch.zeros([h, w, p-1, s], device=device, dtype=torch.float32)
    zs = torch.zeros([h, w, p, s-1], device=device, dtype=torch.float32)
    
    # Alpha relates to the Lipschitz constant. 
    # For 4D, alpha >= 8 is generally recommended for stability.
    alpha = 8 
    
    for it in range(iteration):
        # Calculate primal projections (x0h, x0w, x0p, x0s)
        x0h = y0 - dht_4d(zh)
        x0w = y0 - dwt_4d(zw)
        x0p = y0 - dpt_4d(zp)
        x0s = y0 - dst_4d(zs)
        
        # Average the 4 projections to get the new primal estimate
        x0 = (x0h + x0w + x0p + x0s) / 4
        
        # Update dual variables using gradient descent + clipping (soft-thresholding)
        zh = clip(zh + (1/alpha) * dh_4d(x0), lamda/2)
        zw = clip(zw + (1/alpha) * dw_4d(x0), lamda/2)
        zp = clip(zp + (1/alpha) * dp_4d(x0), lamda/2)
        zs = clip(zs + (1/alpha) * ds_4d(x0), lamda/2)
        
    return x0