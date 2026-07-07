                                                  

import os
import time
import shutil

import torch

import Helpers as helpers
from Utils.Image_Metrics import compute_psnr, compute_ssim, compute_lpips


def hut_estimator(NO_test, v, inp, t):
    batch_size = inp.shape[0]

    def fn(x):
        x = x.reshape(batch_size * NO_test, inp.shape[1], inp.shape[2],
                      inp.shape[3])
        return v(
            x,
            torch.tensor(
                [t]).repeat(
                x.shape[0]).to('cuda')).reshape(
            NO_test,
            batch_size,
            inp.shape[1],
            inp.shape[2],
            inp.shape[3])

    inp_new = inp.repeat(NO_test, 1, 1, 1, 1).clone()
                                            
                                                                                
    eps = ((torch.rand(NO_test, batch_size,
                       inp.shape[1], inp.shape[2], inp.shape[3], device='cuda') < 0.5)) * 2 - 1
                          
    prod = torch.autograd.functional.jvp(
        fn, inp_new, eps, create_graph=True)[1]

    prod = (prod * eps).sum(dim=(2, 3, 4)).mean(0)
    return prod


class FlowPriors(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method
        self.N = args.N

    def model_forward(self, x, t):
        return self.model(x, t)

    def solve_ip(self, test_loader, degradation):
        torch.cuda.empty_cache()

        N, K = self.args.N, self.args.K
        lmbda = self.args.lmbda
        eta = self.args.eta

        H, H_adj = degradation.H, degradation.H_adj
        psnrs, ssims, lpips = [], [], []

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            self.args.batch = batch

            (clean_img, labels) = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            clean_img = clean_img.to('cpu')

                                                           
            x_init = torch.randn(clean_img.shape).to(
                self.device)

            x = x_init.clone()
            x.requires_grad_(True)

            if self.args.start_time > 0.0:
                eps = 1 * self.args.start_time
                dt = (1 - eps) / N
            else:
                         
                dt = 1./N
                eps = 1e-3

            for iteration in range(N):
                num_t = iteration / N * (1 - eps) + eps
                t1 = torch.ones(len(x), device=self.device) * num_t
                t = t1.view(-1, 1, 1, 1)

                x = x.detach().clone()
                x.requires_grad = True
                optim_img = torch.optim.Adam([x], lr=eta)

                if iteration == 0:
                    for k in range(K):
                        x_next = x + self.model_forward(x, t1) * dt

                        y_next = (t + dt) * noisy_img + (1-(t+dt)) * H(x_init)
                        trace_term = hut_estimator(1, self.model_forward, x,  num_t)

                        loss = lmbda * torch.sum((H(x_next) - y_next) ** 2, dim=(1, 2, 3))
                        loss += 0.5 * torch.sum(x ** 2, dim=(1, 2, 3)) + trace_term * dt
                        loss = loss.sum()

                        optim_img.zero_grad()
                        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
                        x.grad = grad
                        optim_img.step()
                else:
                    for k in range(K):
                        pred = self.model_forward(x, t1)
                        x_next = x + pred * dt
                        y_next = (t + dt) * noisy_img + (1-(t+dt)) * H(x_init)

                        trace_term = hut_estimator(
                            1,  self.model_forward, x, num_t)
                        loss = lmbda * torch.sum((H(x_next) - y_next) ** 2, dim=(1, 2, 3))
                        loss += trace_term * dt
                        loss = loss.sum()

                        optim_img.zero_grad()
                        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
                        grad_xt_lik = - 1 / (1-num_t) * (-x + num_t * pred)
                        x.grad = grad + grad_xt_lik
                        optim_img.step()

                x = x + self.model_forward(x, t1) * dt
                torch.cuda.empty_cache()

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
            del restored_img

            del noisy_img, clean_img, x
            torch.cuda.empty_cache()

        return psnrs, ssims, lpips

    def run_method(self, data_loaders, degradation, sigma_noise):
        print(f'Params: start_time={self.args.start_time}, '
              f'lmbda={self.args.lmbda}, '
              f'eta={self.args.eta}\n')

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
                    f'Params: start_time={self.args.start_time}, '
                    f'lmbda={self.args.lmbda}, '
                    f'eta={self.args.eta}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')
