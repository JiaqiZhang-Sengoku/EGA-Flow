                                                  

import os
import time
import shutil

import torch

import Helpers as helpers
from Utils.Image_Metrics import compute_psnr, compute_ssim, compute_lpips


class PnPFlow(object):
    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def model_forward(self, x, t):
        return self.model(x, t)

    def learning_rate_strat(self, lr, t):
        t = t.view(-1, 1, 1, 1)
        gamma_styles = {
            '1_minus_t': lambda lr, t: lr * (1 - t),
            'sqrt_1_minus_t': lambda lr, t: lr * torch.sqrt(1 - t),
            'constant': lambda lr, t: lr,
            'alpha_1_minus_t': lambda lr, t: lr * (1 - t)**self.args.alpha,
        }
        return gamma_styles.get(self.args.gamma_style, lambda lr, t: lr)(lr, t)

    def grad_datafit(self, x, y, H, H_adj):
        return H_adj(H(x) - y) / (self.args.sigma_noise**2)

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def denoiser(self, x, t):
        v = self.model_forward(x, t)
        return x + (1 - t.view(-1, 1, 1, 1)) * v

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj

        if self.args.sigma_noise == 0:
            self.args.sigma_noise = 0.000001

        num_samples = self.args.num_samples
        steps, delta = self.args.steps_pnp, 1 / self.args.steps_pnp
        lr = self.args.sigma_noise**2 * self.args.lr_pnp

        psnrs, ssims, lpips = [], [], []

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            self.args.batch = batch

            (clean_img, labels) = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img, clean_img = noisy_img.to(self.device), clean_img.to('cpu')

                                                            
            x = H_adj(torch.ones_like(noisy_img)).to(self.device)

            with torch.no_grad():
                for count, iteration in enumerate(range(int(steps))):
                    t1 = torch.ones(len(x), device=self.device) * delta * iteration
                    lr_t = self.learning_rate_strat(lr, t1)

                    z = x - lr_t * self.grad_datafit(x, noisy_img, H, H_adj)

                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples):
                        z_tilde = self.interpolation_step(z, t1.view(-1, 1, 1, 1))
                        x_new += self.denoiser(z_tilde, t1)

                    x_new /= num_samples
                    x = x_new

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
        print(f'Params: steps_pnp={self.args.steps_pnp}, '
                f'alpha={self.args.alpha}\n')

        self.args.sigma_noise = sigma_noise

                                          
        files_to_copy = [
            os.path.join(self.args.root, 'Config', 'Method_Config', f'{getattr(self.args, "method_config_stem", self.args.method)}.yaml'),
            os.path.join(self.args.root, 'Src', 'Methods', f'{getattr(self.args, "method_config_stem", self.args.method)}.py'),
        ]

        for f in files_to_copy:
            if os.path.isfile(f):
                shutil.copy2(f, self.args.save_path_ip)

                               
        start = time.time()
        psnrs, ssims, lpips = self.solve_ip(data_loaders[self.args.eval_split], degradation)
        total_time = round(time.time() - start, 4)

                         
        if self.args.compute_metrics:
            avg_psnr = sum(psnrs) / len(psnrs)
            avg_ssim = sum(ssims) / len(ssims)
            avg_lpips = sum(lpips) / len(lpips)

            print(f"Total time = {total_time:.4f}")
            print(f"Average PSNR = {avg_psnr:.4f}")
            print(f"Average SSIM = {avg_ssim:.4f}")
            print(f"Average LPIPS = {avg_lpips:.4f}")

                                     
            eval_file = os.path.join(self.args.save_path_ip, 'eval.txt')
            with open(eval_file, 'a') as file:
                file.write(
                    f'Params: steps_pnp={self.args.steps_pnp}, '
                    f'alpha={self.args.alpha}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
