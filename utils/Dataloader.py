import torch
import numpy as np
import scipy.io as sio
import math

def load_data(input_path, mask_path, device):
    """
    Loads image and mask data from .mat files, expands the mask to match
    image dimensions (tiling), and generates the compressed measurement 'y'.
    Now detects keys automatically and normalizes im_orig based on its max value.
    """
    # 1. Load data from MATLAB files
    mat_data = sio.loadmat(input_path)
    mask_data = sio.loadmat(mask_path)
    
    # 2. Process im_orig (Input datacube)
    # Automatically find the data key (ignoring MATLAB internal keys like __header__)
    im_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if not im_keys:
        raise KeyError(f"No valid data variable found in {input_path}")
    
    # Load the first available variable and move to device
    im_orig = torch.from_numpy(mat_data[im_keys[0]]).float().to(device)
    
    # Dynamic Normalization: Scale to [0, 1] based on the file's local maximum
    max_val = im_orig.max()
    if max_val > 0:
        im_orig = im_orig / max_val
    else:
        print(f"[Warning] {input_path} is empty or all zeros.")

    H, W, NF = im_orig.shape 
    
    # 3. Process mask (Using the same automatic key detection logic)
    mask_keys = [k for k in mask_data.keys() if not k.startswith('__')]
    mask_small = torch.from_numpy(mask_data[mask_keys[0]]).float().to(device)
    h_m, w_m, _ = mask_small.shape
    
    # 4. Mask Expansion (Tiling logic)
    n_rep_h = math.ceil(H / h_m)
    n_rep_w = math.ceil(W / w_m)
    mask_expanded = mask_small.repeat(n_rep_h, n_rep_w, 1)
    
    # 5. Cropping
    mask_final = mask_expanded[:H, :W, :]
    
    # 6. Generate measurement 'y'
    y = torch.sum(im_orig * mask_final, dim=2)
    
    return y, im_orig, mask_final

def load_image_only(path, device):
    """
    Loads image data from a .mat file by detecting the data key,
    and normalizes the values to the [0, 1] range based on the file's max value.
    """
    mat_data = sio.loadmat(path)
    
    # Filter MATLAB metadata keys to find the actual data variable
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    if not data_keys:
        raise KeyError(f"No data variable found in {path}")
    
    # Select the first data key
    variable_name = data_keys[0]
    
    # Convert to PyTorch tensor
    img = torch.from_numpy(mat_data[variable_name]).float().to(device)
    
    # Dynamic Normalization: Scale based on the maximum value found in the tensor
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    else:
        print(f"[Warning] File {os.path.basename(path)} is empty or all zeros.")
    
    return img

def prepare_global_mask(mask_path, H, W, device):
    """
    Loads the small mask, tiles it to match image dimensions, 
    and pre-calculates the mask sum for the GAP optimizer.
    """
    mask_data = sio.loadmat(mask_path)
    # mask_small corresponds to 'best_C' in MATLAB (h_m, w_m, NF)
    mask_small = torch.from_numpy(mask_data['C']).float().to(device)
    h_m, w_m, _ = mask_small.shape
    
    # Calculate repetitions needed to cover the full image size (H, W)
    n_rep_h = math.ceil(H / h_m)
    n_rep_w = math.ceil(W / w_m)
    
    # Expand the mask using tiling logic (equivalent to kron with ones in MATLAB)
    mask_expanded = mask_small.repeat(n_rep_h, n_rep_w, 1)
    
    # Crop to exact image dimensions
    mask_final = mask_expanded[:H, :W, :]
    
    # Pre-calculate the squared sum across the spectral dimension for GAP steps
    mask_sum = torch.sum(mask_final**2, dim=2)
    mask_sum[mask_sum == 0] = 1  # Prevent division by zero
    
    return mask_final, mask_sum

import torch
import scipy.io as sio
import math

def prepare_global_mask4D(mask_path, H, W, P, S, device):
    """
    Loads a 4D mask [h_m, w_m, p_m, s_m], tiles it spatially to match 
    the target [H, W, P, S], and calculates the normalization factor for GAP.
    
    Args:
        mask_path: Path to the .mat file containing the mask 'C'.
        H, W: Target spatial dimensions.
        P, S: Target Polarization and Spectral dimensions.
        device: Torch device (CPU/CUDA).
    """
    mask_data = sio.loadmat(mask_path)
    
    # Load mask and convert to tensor. 
    # Expected shape from .mat: [h_m, w_m, p_m, s_m]
    mask_small = torch.from_numpy(mask_data['C']).float().to(device)
    h_m, w_m, p_m, s_m = mask_small.shape
    
    # 1. Calculate spatial repetitions needed for H and W
    n_rep_h = math.ceil(H / h_m)
    n_rep_w = math.ceil(W / w_m)
    
    # 2. Check if Polarization and Spectral dimensions match
    # If they don't match, you might need tiling or interpolation logic here.
    # Assuming the mask provided already matches the P and S of the sensor.
    n_rep_p = 1
    n_rep_s = 1
    
    # 3. Perform 4D Tiling
    # .repeat(H_reps, W_reps, P_reps, S_reps)
    mask_expanded = mask_small.repeat(n_rep_h, n_rep_w, n_rep_p, n_rep_s)
    
    # 4. Crop to exact target dimensions [H, W, P, S]
    mask_final = mask_expanded[:H, :W, :, :]
    
    # 5. Pre-calculate the squared sum across the P and S dimensions
    # For GAP, the denominator is the sum of squares of the mask 
    # across all dimensions being compressed (dims 2 and 3).
    mask_sum = torch.sum(mask_final**2, dim=(2, 3))
    
    # Prevent division by zero for pixels where the mask might be all zeros
    mask_sum[mask_sum == 0] = 1 
    
    return mask_final, mask_sum