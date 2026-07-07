                                                  

import lpips
import torch
from ignite.metrics import SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torchvision.transforms as v2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)                       


def postprocess(img, args):
    if args.dataset == "afhq_cat":
        img = (img + 1) / 2
    else:
        inv_trans = v2.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1./0.5, 1./0.5, 1./0.5])
        img = inv_trans(img)

    return img


def compute_psnr(clean_img, noisy_img, rec_img, args, H_adj):
                                                                                
    clean_img = postprocess(clean_img.clone(), args)
    noisy_img = postprocess(noisy_img.clone(), args)
    rec_img = postprocess(rec_img.clone(), args)
    H_adj_noisy_img = postprocess(H_adj(noisy_img), args)

    clean_img = clean_img.permute(0, 2, 3, 1).cpu().data.numpy()
    if args.problem == 'superresolution':
        noisy_img = H_adj_noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    else:
        noisy_img = noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    rec_img = rec_img.permute(0, 2, 3, 1).cpu().data.numpy()

                         
    psnr_rec = PSNR(clean_img, rec_img, data_range=1.0)
    psnr_noisy = PSNR(clean_img, noisy_img, data_range=1.0)

    return psnr_rec, psnr_noisy


def compute_lpips(clean_img, noisy_img, rec_img, args, H_adj):
                                                                                 
    clean_img = postprocess(clean_img.clone(), args)
    noisy_img = postprocess(noisy_img.clone(), args)
    rec_img = postprocess(rec_img.clone(), args)
    H_adj_noisy_img = postprocess(H_adj(noisy_img), args)

                                                                  
    clean_img = clean_img.to(DEVICE)
    rec_img = rec_img.to(DEVICE)

    if args.problem == 'superresolution':
        noisy_img = H_adj_noisy_img.to(DEVICE)
    else:
        noisy_img = noisy_img.to(DEVICE)

                                                                        
    lpips_rec = loss_fn_alex(clean_img, rec_img, normalize=True).mean().item()
    lpips_noisy = loss_fn_alex(
        clean_img, noisy_img, normalize=True).mean().item()

    return lpips_rec, lpips_noisy


def compute_ssim(clean_img, noisy_img, rec_img, args, H_adj):
                                                                                
    H_adj_noisy_img = postprocess(H_adj(noisy_img), args).cpu()
    clean_img = postprocess(clean_img.clone(), args).cpu()
    noisy_img = postprocess(noisy_img.clone(), args).cpu()
    rec_img = postprocess(rec_img.clone(), args).cpu()

                                                                   
    if args.problem == 'superresolution':
        noisy_img = H_adj_noisy_img
    else:
        noisy_img = noisy_img

                                                          
    ssim_metric = SSIM(data_range=1.0)
    ssim_metric_noisy = SSIM(data_range=1.0)

                         
    ssim_metric.update((rec_img, clean_img))
    ssim_rec = ssim_metric.compute()
    ssim_metric_noisy.update((noisy_img, clean_img))
    ssim_noisy = ssim_metric_noisy.compute()

    return ssim_rec, ssim_noisy
