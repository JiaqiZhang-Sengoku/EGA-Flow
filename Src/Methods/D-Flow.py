                                                  

import os
import time
import shutil

import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

import Helpers as helpers
from Utils.Image_Metrics import compute_psnr, compute_ssim, compute_lpips


class DFlow(object):

\
\
       

    def __init__(self, model, device, args, mask_type=None):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method
        self.mask_type = mask_type

    def model_forward(self, x, t):
        return self.model(x, t)

    def gaussian(self, img):
        if img.ndim != 4:
            raise RuntimeError(
                f"Expected input `img` to be an 4D tensor, but got {img.shape}")
        return (img**2).sum([1, 2, 3]) * 0.5

    def forward_flow_matching(self, z):
        steps = self.args.steps_euler
        delta = (1 - self.args.start_time) / (steps - 1)
        for i in range(steps - 1):
            t1 = torch.ones(len(z), device=self.device) * delta * i + self.args.start_time
            z = z + delta * self.model_forward(z + delta /
                                               2 * self.model_forward(z, t1), t1 + delta / 2)
        return z

    def inverse_flow_matching(self, z):
        flow_class = cnf(self.model, self.args.model_type)
        z_t = odeint(flow_class, z,
                     torch.tensor([1.0, 0.0]).to(self.device),
                     atol=1e-5,
                     rtol=1e-5,
                     method='dopri5',
                     )
        x = z_t[-1].detach()
        return x

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            self.args.batch = batch

            (clean_img, labels) = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')

            x = H_adj(noisy_img.clone()).to(self.device)
            z = self.inverse_flow_matching(x).to(self.device)

                                                        
            z = np.sqrt(self.args.alpha) * z + np.sqrt(1 -
                                                       self.args.alpha) * torch.randn_like(z)
            z = z.detach().requires_grad_(True)

                                        
            optim_img = torch.optim.LBFGS(
                [z], max_iter=self.args.LBFGS_iter, history_size=100, line_search_fn='strong_wolfe')
            d = z.shape[1] * z.shape[2] * z.shape[3]

                                                               
            for iteration in range(self.args.max_iter):
                def closure():
                    optim_img.zero_grad()                   
                    reg = - torch.clamp(self.gaussian(z), min=-1e6, max=1e6) + (
                        d - 1) * torch.log(torch.sqrt(torch.sum(z**2, dim=(1, 2, 3))) + 1e-5)

                    loss = (torch.sum((H(self.forward_flow_matching(z)) -
                            noisy_img)**2, dim=(1, 2, 3)) + self.args.lmbda * reg).sum()
                    loss.backward()                     
                    return loss

                optim_img.step(closure)

                restored_img = self.forward_flow_matching(z.detach())
                del restored_img

            z = z.detach().requires_grad_(False)
            restored_img = self.forward_flow_matching(z.detach())

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

            del restored_img, noisy_img, clean_img, x, z

        return psnrs, ssims, lpips

    def run_method(self, data_loaders, degradation, sigma_noise):
        print(f'Params: lmbda={self.args.lmbda}, '
              f'alpha={self.args.alpha}, '
              f'LBFGS_iter={self.args.LBFGS_iter}\n')

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
                    f'Params: lmbda={self.args.lmbda}, '
                    f'alpha={self.args.alpha}, '
                    f'LBFGS_iter={self.args.LBFGS_iter}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')


class cnf(torch.nn.Module):

    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def model_forward(self, x, t):
        return self.model(x, t)

    def forward(self, t, x):
        with torch.no_grad():
            z = self.model_forward(x, t.repeat(x.shape[0]))
        return z
