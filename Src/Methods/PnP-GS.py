import os
import shutil
import time

import torch

import Helpers as helpers
from Utils.Image_Metrics import compute_lpips, compute_psnr, compute_ssim


class GradientStepDenoiser(object):
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.grad_matching = True
        self.sigma_step = False
        self.weight_Ds = 1.0

    def calculate_grad(self, x, sigma, compute_g=False):
        x = x.detach().float().requires_grad_(True)
        sigma = sigma.reshape(-1).to(x.device).float()
        denoised = self.model(x, sigma)
        grad_outputs = x - denoised
        jacobian_term = torch.autograd.grad(
            denoised,
            x,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        grad = x - denoised - jacobian_term
        if compute_g:
            g = 0.5 * torch.sum((x - denoised).reshape((x.shape[0], -1)) ** 2)
            return grad, denoised, g.detach()
        return grad, denoised

    def forward(self, x, sigma):
        grad, _ = self.calculate_grad(x, sigma)
        if self.sigma_step:
            return x - self.weight_Ds * sigma.view(-1, 1, 1, 1) * grad, grad
        return x - self.weight_Ds * grad, grad


class PnPGS(object):
    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = GradientStepDenoiser(model, device, args)

    def grad_datafit(self, x, y, H, H_adj):
        return H_adj(H(x) - y) / (self.args.sigma_noise ** 2)

    def prox_datafit(self, x, y, H, H_adj):
        if self.args.problem == 'random_inpainting':
            return H(y) - H(x) + x
        if self.args.problem == 'box_inpainting':
            mask = H(torch.ones_like(x))
            return mask * y + (1 - mask) * x
        return x

    def initial_point(self, clean_img, noisy_img, degradation):
        if self.args.problem == 'random_inpainting':
            return 1.5 * noisy_img.clone() - degradation.H(noisy_img)
        return degradation.H_adj(noisy_img.clone()).to(self.device)

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj
        sigma_noise = self.args.sigma_noise
        lr = sigma_noise ** 2 * self.args.lr_pnp
        max_iter = self.args.max_iter
        alpha = self.args.alpha
        sigma_factor = self.args.sigma_factor
        psnrs, ssims, lpips = [], [], []
        loader = iter(test_loader)

        for batch in range(self.args.max_batch):
            self.args.batch = batch
            clean_img, _ = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * sigma_noise
            noisy_img, clean_img = noisy_img.to(self.device), clean_img.to('cpu')
            x = self.initial_point(clean_img, noisy_img, degradation)

            for iteration in range(int(max_iter)):
                x_old = x.detach()
                if self.args.algo == 'hqs' and self.args.problem in {'random_inpainting', 'box_inpainting'}:
                    sigma_level = sigma_factor * sigma_noise * torch.ones(len(x), device=self.device)
                    torch.set_grad_enabled(True)
                    grad_reg, _ = self.model.calculate_grad(x_old, sigma_level)
                    torch.set_grad_enabled(False)
                    denoised = (x_old - grad_reg).detach()
                    x = self.prox_datafit(denoised, noisy_img, H, H_adj).detach()
                else:
                    if self.args.problem != 'denoising':
                        z = x_old - lr * self.grad_datafit(x_old, noisy_img, H, H_adj)
                    else:
                        z = x_old
                    sigma_level = sigma_factor * sigma_noise * torch.ones(len(x), device=self.device)
                    torch.set_grad_enabled(True)
                    grad_reg, _ = self.model.calculate_grad(z, sigma_level)
                    torch.set_grad_enabled(False)
                    denoised = (z - grad_reg).detach()
                    x = ((1 - alpha) * z + alpha * denoised).detach()

            restored_img = x.detach().clone()
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
        print(
            f'Params: algo={self.args.algo}, max_iter={self.args.max_iter}, '
            f'lr_pnp={self.args.lr_pnp}, alpha={self.args.alpha}, sigma_factor={self.args.sigma_factor}\n'
        )
        self.args.sigma_noise = sigma_noise if sigma_noise > 0 else 1e-6

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
                file.write(
                    f'Params: algo={self.args.algo}, max_iter={self.args.max_iter}, '
                    f'lr_pnp={self.args.lr_pnp}, alpha={self.args.alpha}, sigma_factor={self.args.sigma_factor}\n'
                    '---------------------------------------------------------\n'
                )
                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')
                file.write('---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
                file.write(f'Peak memory MB = {peak_memory_mb:.2f}\n')
