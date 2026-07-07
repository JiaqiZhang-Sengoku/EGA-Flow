import os
import shutil
import time

import torch

import Helpers as helpers
from Utils.Image_Metrics import compute_lpips, compute_psnr, compute_ssim


class Degraded(object):
    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []
        loader = iter(test_loader)

        for batch in range(self.args.max_batch):
            self.args.batch = batch
            clean_img, _ = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img, clean_img = noisy_img.to(self.device), clean_img.to('cpu')

            restored_img = H_adj(noisy_img).detach().clone()

            if self.args.compute_metrics:
                psnr_rec, psnr_noisy = compute_psnr(clean_img, noisy_img, restored_img, self.args, H_adj)
                print(f"Batch {batch}: psnr_rec={psnr_rec}, psnr_noisy={psnr_noisy}")
                psnrs.append(psnr_rec)
                ssim_rec, ssim_noisy = compute_ssim(clean_img, noisy_img, restored_img, self.args, H_adj)
                print(f"Batch {batch}: ssim_rec={ssim_rec}, ssim_noisy={ssim_noisy}")
                ssims.append(ssim_rec)
                lpip_rec, lpip_noisy = compute_lpips(clean_img, noisy_img, restored_img, self.args, H_adj)
                lpips.append(lpip_rec)

            helpers.save_images(clean_img, noisy_img, restored_img, self.args, H_adj)

        return psnrs, ssims, lpips

    def run_method(self, data_loaders, degradation, sigma_noise):
        print('Params: direct degraded observation baseline\n')
        self.args.sigma_noise = sigma_noise

        files_to_copy = [
            os.path.join(self.args.root, 'Config', 'Method_Config', f'{getattr(self.args, "method_config_stem", self.args.method)}.yaml'),
            os.path.join(self.args.root, 'Src', 'Methods', f'{getattr(self.args, "method_config_stem", self.args.method)}.py'),
        ]
        for f in files_to_copy:
            if os.path.isfile(f):
                shutil.copy2(f, self.args.save_path_ip)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.time()
        psnrs, ssims, lpips = self.solve_ip(data_loaders[self.args.eval_split], degradation)
        total_time = round(time.time() - start, 4)
        peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

        if self.args.compute_metrics:
            avg_psnr = sum(psnrs) / len(psnrs)
            avg_ssim = sum(ssims) / len(ssims)
            avg_lpips = sum(lpips) / len(lpips)

            print(f"Total time = {total_time:.4f}")
            print(f"Average PSNR = {avg_psnr:.4f}")
            print(f"Average SSIM = {avg_ssim:.4f}")
            print(f"Average LPIPS = {avg_lpips:.4f}")
            print(f"Peak memory MB = {peak_memory_mb:.2f}")

            eval_file = os.path.join(self.args.save_path_ip, 'eval.txt')
            with open(eval_file, 'a') as file:
                file.write('Params: direct degraded observation baseline\n')
                file.write('---------------------------------------------------------\n')
                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')
                file.write('---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
                file.write(f'Peak memory MB = {peak_memory_mb:.2f}\n')
