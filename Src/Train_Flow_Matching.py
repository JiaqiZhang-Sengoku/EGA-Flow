                                                  

                   
 
                                                                                                      
                                                                                           
                                  
                                                        
 
                                                               
                                                                                   
                                  
                                                                                                  

import torch
from tqdm import tqdm
import numpy as np
import Helpers as helpers
import os
import ot
from torchdiffeq import odeint_adjoint as odeint
import time


def get_dataset_display_name(dataset):
    names = {
        'celeba': 'CelebA',
        'afhq_cat': 'AFHQ-Cat',
    }
    return names.get(str(dataset), str(dataset))


class FLOW_MATCHING(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)
        self.img_size = 128

    def train_FM_model(self, train_loader, opt, num_epoch):
        print("train FM model")
        tq = tqdm(range(num_epoch), desc='loss')
        for ep in tq:
            for iteration, (x, labels) in enumerate(train_loader):

                if x.size(0) == 0:
                    continue
                x = x.to(self.device)
                z = torch.randn(
                    x.shape[0],
                    self.num_channels,
                    self.d,
                    self.d,
                    device=self.device,
                    requires_grad=True)
                
                t1 = torch.rand(x.shape[0], 1, 1, 1, device=self.device)

                                  
                x0 = z.clone()         
                self.img_size = x0.shape[-1]
                x1 = x.clone()               
                a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)

                M = ot.dist(x0.view(len(x0), -1).cpu().data.numpy(),
                            x1.view(len(x1), -1).cpu().data.numpy())
                plan = ot.emd(a, b, M)
                p = plan.flatten()
                p = p / p.sum()
                choices = np.random.choice(
                    plan.shape[0] * plan.shape[1], p=p, size=len(x0), replace=True)
                i, j = np.divmod(choices, plan.shape[1])
                x0 = x0[i]         
                x1 = x1[j]         
                xt = t1 * x1 + (1 - t1) * x0
                loss = torch.sum(
                    (self.model(xt, t1.squeeze()) - (x1 - x0))**2) / x.shape[0]
                opt.zero_grad()
                loss.backward()
                opt.step()

                                       
                with open(os.path.join(self.args.save_path, 'loss_training.txt'), 'a') as file:
                    file.write(
                        f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

                                                                       
                                     
            if ep % 5 == 0:
                             
                for i in range(0, 4):
                    gen_img = self.apply_flow_matching(1)
                    helpers.save_image(gen_img[0].unsqueeze(0), self.args.save_path,
                                               f'gen_img_ep_{ep}_{i}.png')

            if ep % 20 == 0:
                            
                print(f"Saving the model at epoch {ep}...")
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.save_path, 'model_{}.pt'.format(ep)))                    

                              
                                                         
                                                                          
                                                                   

    def apply_flow_matching(self, NO_samples):
        self.model.eval()
        with torch.no_grad():
            model_class = cnf(self.model)
            latent = torch.randn(
                NO_samples,
                self.num_channels,
                self.d,
                self.d,
                device=self.device,
                requires_grad=False)
            z_t = odeint(model_class, latent,
                         torch.tensor([0.0, 1.0]).to(self.device),
                         atol=1e-5,
                         rtol=1e-5,
                         method='dopri5',
                         )
            x = z_t[-1].detach()
        self.model.train()
        return x

    def sample_plot(self, x, ep=None):
        try:
            os.makedirs(self.args.save_path + 'Results_Samplings/')
        except BaseException:
            pass

        reco = helpers.postprocess(self.apply_flow_matching(16), self.args)
        helpers.save_images(reco, x[:16], self.args.save_path + 'Results_Samplings/' +
                         'samplings_ep_{}'.format(ep), self.args)

                                                    
        if ep == 0:
            gt = x[:16]
            gt = helpers.postprocess(gt, self.args)
            helpers.save_image(gt, gt, self.args.save_path + 'Results_Samplings/' +
                             'train_samples_ep_{}'.format(ep), self.args)

    def train(self, data_loaders):
        print("train")
        self.args.save_path = os.path.join(self.args.root, 'Trained_Models', self.args.dataset, self.args.model_type,
                                           time.strftime("%Y%m%d-%H%M%S"))

                                                                                           
        try:
            os.makedirs(self.args.save_path)
        except BaseException:
            pass

        self.model_path = os.path.join(self.args.root, 'Model_Checkpoints')
        try:
            os.makedirs(self.model_path)
        except BaseException:
            pass

                    
        train_loader = data_loaders['train']

                                                                 
        with open(os.path.join(self.args.save_path, 'model_info.txt'), 'w') as file:
            file.write(f'PARAMETERS\n')
            file.write(
                f'Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n')
            file.write(f'Number of epochs: {self.args.num_epoch}\n')
            file.write(f'Batch size: {self.args.batch_size_train}\n')
            file.write(f'Learning rate: {self.lr}\n')

                        
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_FM_model(train_loader, opt, num_epoch=self.args.num_epoch)

                          
        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{get_dataset_display_name(self.args.dataset)}.pt'))


class cnf(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.no_grad():
                                            
            z = self.model(x, t.repeat(x.shape[0]))
        return z
