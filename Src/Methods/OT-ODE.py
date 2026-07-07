                                                  

import os
import time
import shutil

import torch

import Helpers as helpers
from Utils.Image_Metrics import compute_psnr, compute_ssim, compute_lpips


class OTOde(object):
    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def model_forward(self, x, t):
        return self.model(x, t)

    def initialization(self, noisy_img, t0):
        return t0 * noisy_img + (1-t0) * torch.randn_like(noisy_img)

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)

            self.args.batch = batch

            noisy_img = H(clean_img.clone().to(
                self.device))                             
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')

                                                            
                                                          
            x = self.initialization(H_adj(noisy_img.clone()), self.args.start_time)

            steps, delta = self.args.steps_ode, 1 / self.args.steps_ode

            for count, iteration in enumerate(range(int(steps * self.args.start_time), int(steps))):
                with ((torch.no_grad())):
                    t1 = torch.ones(len(x), device=self.device) * delta * iteration
                    vt = self.model_forward(x, t1)
                    rt_squared = ((1-t1)**2 / ((1-t1)**2 + t1**2)).view(-1, 1, 1, 1)
                    x1_hat = x + (1-t1.view(-1, 1, 1, 1)) * vt

                                               
                    d = noisy_img - H(x1_hat)
                    sol = torch.zeros_like(d)

                    for i in range(d.shape[0]):
                        sol_tmp = 1 / (H(torch.ones_like(x))[i] * rt_squared[i] + self.args.sigma_noise**2) * d[i]
                        sol[i] = sol_tmp.reshape(d[i].shape)

                    vec = H_adj(sol)

                                            
                t = t1.view(-1, 1, 1, 1)
                if self.args.gamma == "constant":
                    gamma = 1
                elif self.args.gamma == "gamma_t":
                    gamma = torch.sqrt(t / (t**2 + (1 - t)**2))
                g = torch.autograd.functional.vjp(lambda z:  self.model_forward(z, t1), inputs=x, v=vec)[1]

                with torch.no_grad():
                    g = vec + (1-t1.view(-1, 1, 1, 1)) * g
                    ratio = (1-t1.view(-1, 1, 1, 1)) / t1.view(-1, 1, 1, 1)
                    v_adapted = vt + ratio * gamma * g
                    x_new = x + delta * v_adapted
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
        print(f'Params: steps_ode={self.args.steps_ode}, '
              f'start_time={self.args.start_time}, '
              f'gamma={self.args.gamma}\n')

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
                    f'Params: steps_ode={self.args.steps_ode}, '
                    f'start_time={self.args.start_time}, '
                    f'gamma={self.args.gamma}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
