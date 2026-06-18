import os
import torch
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import time
import argparse
import math

# --- Local imports from your project structure ---
from utils.utils import psnr, ssim, calculate_sam
from model.TV_denoising import TV_denoising3d
from utils.Dataloader import load_image_only, prepare_global_mask

# Configuration and arguments
parser = argparse.ArgumentParser(description="GAP-TV Reconstruction for Compressive Sensing")
parser.add_argument('--device', default='0', help='GPU device index')
parser.add_argument('--input_dir', default='./Cave1', help='Directory with input .mat files')
parser.add_argument('--output_dir', default='./Test', help='Directory to save results')
#parser.add_argument('--mask_path', default='./SSP/Bands_31_Mux_1.mat', help='Path to the small mask .mat file')
parser.add_argument('--mask_path', default='./SSP/Bands_4_12_Mux_1.mat', help='Path to the small mask .mat file')
parser.add_argument('--iters', default=300, type=int, help='Number of optimization iterations')
args = parser.parse_args()

# Set computing device (CUDA/CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda")
print(f'Using GPU: {args.device}')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def process_file(input_path, output_path, mask, mask_sum):
    """Performs GAP-TV reconstruction on a single image cube."""
    im_orig = load_image_only(input_path, device)
    
    # Generate the compressed measurement 'y' (Simulation of the sensing process)
    y = torch.sum(im_orig * mask, dim=2)
    
    # Initialize the estimate 'x' (Back-projection)
    x = y.unsqueeze(2).expand_as(mask) * mask
    y1 = torch.zeros_like(y, device=device)

    # GAP-TV Optimization Loop
    for i in tqdm(range(args.iters), desc=f"Processing {os.path.basename(input_path)}", leave=False):
        # Euclidean projection step (GAP)
        yb = torch.sum(mask * x, dim=2)
        y1 = y1 + (y - yb)
        temp = (y1 - yb) / mask_sum
        x = x + (temp.unsqueeze(2).expand_as(mask) * mask)
        
        # Regularization step (Total Variation Denoising)
        x = TV_denoising3d(x, 5, 5).clamp(0, 1)

    # Ensure final values are strictly within [0, 1]
    x.clamp_(0.0, 1.0)
    
    # Calculate performance metrics
    avg_psnr = np.mean([psnr(x[..., k], im_orig[..., k]) for k in range(im_orig.shape[2])])
    avg_ssim = np.mean([ssim(x[..., k], im_orig[..., k]) for k in range(im_orig.shape[2])])
    avg_sam  = calculate_sam(x, im_orig) 
    
    # Save the reconstructed cube and metrics to a .mat file
    sio.savemat(output_path, {
        'av_psnr': avg_psnr, 
        'av_ssim': avg_ssim, 
        'av_sam': avg_sam, 
        'reconstructed': x.cpu().numpy()
    })
    
    return avg_psnr, avg_ssim, avg_sam

def run():
    """Orchestrates the processing of the entire dataset."""
    metrics = {'psnr': [], 'ssim': [], 'sam': []}
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.mat')]
    
    if not files:
        print(f"No .mat files found in {args.input_dir}")
        return

    # --- Initialization Phase (Run once) ---
    # Load first image to detect dimensions
    first_im = load_image_only(os.path.join(args.input_dir, files[0]), device)
    H, W, _ = first_im.shape
    
    print(f"[*] Resolution detected: {H}x{W}. Initializing global mask...")
    mask_global, mask_sum_global = prepare_global_mask(args.mask_path, H, W, device)
    # --------------------------------------

    # Iterate through each file in the directory
    for filename in files:
        input_file = os.path.join(args.input_dir, filename)
        output_file = os.path.join(args.output_dir, 'proc_' + filename)
        
        p, s, sam = process_file(input_file, output_file, mask_global, mask_sum_global)
        
        metrics['psnr'].append(p)
        metrics['ssim'].append(s)
        metrics['sam'].append(sam)

    # Display final dataset statistics
    if metrics['psnr']:
        print("\n" + "="*45)
        print(f"FINAL DATASET METRICS ({len(files)} cubes)")
        print(f"Average PSNR: {np.mean(metrics['psnr']):.4f} dB")
        print(f"Average SSIM: {np.mean(metrics['ssim']):.4f}")
        print(f"Average SAM:  {np.mean(metrics['sam']):.4f} rad")
        print("="*45)

if __name__ == "__main__":
    # Ensure no gradients are computed to save memory and time
    with torch.no_grad():
        start_time = time.time()
        run()
        print(f'\nTotal execution time: {time.time() - start_time:.2f}s')