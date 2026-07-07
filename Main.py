import argparse
import importlib.util
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from Helpers import define_unet, load_cfg_from_cfg_file, load_model, merge_cfg_from_list
from Src.DataLoaders import DataLoaders
from Src.Train_Flow_Matching import FLOW_MATCHING
from Utils.Degradations import *


torch.cuda.empty_cache()


METHOD_CONFIG_NAMES = {
    'd_flow': 'D-Flow',
    'degraded': 'Degraded',
    'ega_flow': 'EGA-Flow',
    'flow_priors': 'Flow-Priors',
    'ot_ode': 'OT-ODE',
    'pnp_flow': 'PnP-Flow',
    'pnp_gs': 'PnP-GS',
    'restora_flow': 'Restora-Flow',
}

METHOD_CLASS_NAMES = {
    'd_flow': 'DFlow',
    'degraded': 'Degraded',
    'ega_flow': 'EGAFlow',
    'flow_priors': 'FlowPriors',
    'ot_ode': 'OTOde',
    'pnp_flow': 'PnPFlow',
    'pnp_gs': 'PnPGS',
    'restora_flow': 'RestoraFlow',
}

METHOD_ALIASES = {
    **{name: key for key, name in METHOD_CONFIG_NAMES.items()},
    **{name.lower(): key for key, name in METHOD_CONFIG_NAMES.items()},
    'flow-prioris': 'flow_priors',
    'flow_prioris': 'flow_priors',
}

DATASET_DISPLAY_NAMES = {
    'celeba': 'CelebA',
    'afhq_cat': 'AFHQ-Cat',
}


def normalize_method_name(method):
    method = str(method)
    return METHOD_ALIASES.get(method, method)


def get_method_config_stem(method):
    return METHOD_CONFIG_NAMES.get(method, method)


def get_method_config_file(root, method):
    return os.path.join(root, 'Config', 'Method_Config', f'{get_method_config_stem(method)}.yaml')


def get_dataset_display_name(dataset):
    return DATASET_DISPLAY_NAMES.get(str(dataset), str(dataset))


def get_flow_checkpoint_path(args):
    checkpoint_name = getattr(args, 'checkpoint_name', '')
    if checkpoint_name:
        checkpoint_path = checkpoint_name
    else:
        checkpoint_path = os.path.join('Model_Checkpoints', f'{get_dataset_display_name(args.dataset)}.pt')
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(args.root, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    return checkpoint_path


def get_gradient_step_checkpoint_path(args):
    checkpoint_path = getattr(args, 'gradient_step_checkpoint', None)
    if checkpoint_path is None or checkpoint_path == '':
        checkpoint_path = os.path.join('Model_Checkpoints', f'{get_dataset_display_name(args.dataset)}-GradientStep.pt')
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(args.root, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Gradient-step checkpoint not found: {checkpoint_path}')
    return checkpoint_path


def load_method_class(method):
    class_name = METHOD_CLASS_NAMES[method]
    method_file = os.path.join(
        os.path.dirname(__file__),
        'Src',
        'Methods',
        f'{get_method_config_stem(method)}.py',
    )
    spec = importlib.util.spec_from_file_location(f'_ega_flow_method_{method}', method_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load method implementation: {method_file}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_gradient_step_unet(unet, args, device):
    checkpoint_path = get_gradient_step_checkpoint_path(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    unet.load_state_dict(state_dict)
    unet.to(device)
    unet.eval()
    args.gradient_step_checkpoint = checkpoint_path
    return unet


def parse_args(method_name=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('Config/Main_Config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = os.path.join(cfg.root, 'Config', 'Dataset_Config', f'{get_dataset_display_name(cfg.dataset)}.yaml')
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    cfg.method = normalize_method_name(cfg.method)
    if method_name is None:
        method_name = cfg.method
    method_name = normalize_method_name(method_name)
    cfg.method = method_name
    cfg.method_config_stem = get_method_config_stem(method_name)

    method_config_file = get_method_config_file(cfg.root, method_name)
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
        cfg.method = normalize_method_name(cfg.method)
        cfg.method_config_stem = get_method_config_stem(cfg.method)

    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def make_degradation(args, device):
    sigma_noise = args.sigma_y
    if args.problem == 'denoising':
        sigma_noise = float(getattr(args, 'denoise_sigma', 0.2))
        additional_dir_name = f'sigma_{sigma_noise}'
        degradation = Denoising()
    elif args.problem == 'box_inpainting':
        additional_dir_name = f'size_{args.mask_size_x}x{args.mask_size_y}'
        degradation = BoxInpainting((args.mask_size_x, args.mask_size_y))
    elif args.problem == 'random_inpainting':
        additional_dir_name = f'p_{args.p_value}'
        degradation = RandomInpainting(args.p_value)
    elif args.problem == 'superresolution':
        additional_dir_name = f'sf_{args.sf}'
        degradation = Superresolution(args.sf, args.dim_image)
    elif args.problem == 'gaussian_deblurring_FFT':
        sigma_blur = float(getattr(args, 'blur_sigma', 1.0 if args.dim_image == 128 else 3.0))
        kernel_size = int(getattr(args, 'blur_kernel_size', 61))
        additional_dir_name = f'blur_sigma_{sigma_blur}_kernel_{kernel_size}'
        degradation = GaussianDeblurring(
            sigma_blur,
            kernel_size,
            'fft',
            args.num_channels,
            args.dim_image,
            device,
        )
    else:
        raise ValueError(f'Problem not supported: {args.problem}')
    return degradation, sigma_noise, additional_dir_name


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.train:
        print('Training...')
        data_loaders = DataLoaders(
            args.dataset,
            args.batch_size_train,
            args.batch_size_train,
            args.dim_image,
            train=True,
            data_root=getattr(args, 'data_root', None),
        ).load_data()
        unet = define_unet(args, device)
        if args.model_type != 'ot':
            raise ValueError(f'Unsupported model type: {args.model_type}')
        generative_model = FLOW_MATCHING(unet, device, args)
        generative_model.train(data_loaders)
        print('Training done!')

    if args.eval:
        print('Starting evaluation...')
        if args.method == 'degraded':
            model = None
        elif args.method == 'pnp_gs':
            unet = define_unet(args, device)
            model = load_gradient_step_unet(unet, args, device)
        else:
            unet = define_unet(args, device)
            model_checkpoint_path = get_flow_checkpoint_path(args)
            model = load_model(unet, args.model_type, model_checkpoint_path, device)

        degradation, sigma_noise, additional_dir_name = make_degradation(args, device)
        print(f'Solving {args.problem} with {args.method}...')

        data_loaders = DataLoaders(
            args.dataset,
            args.batch_size_ip,
            args.batch_size_ip,
            args.dim_image,
            data_root=getattr(args, 'data_root', None),
        ).load_data()

        timestamp = time.strftime('%Y%m%d-%H%M%S')
        args.save_path_ip = os.path.join(
            args.root,
            'Results',
            args.dataset,
            args.model_type,
            args.problem,
            additional_dir_name,
            get_method_config_stem(args.method),
            args.eval_split,
            timestamp,
        )
        os.makedirs(args.save_path_ip, exist_ok=True)
        print('Output will be saved to ', args.save_path_ip)

        if args.method not in METHOD_CLASS_NAMES:
            raise ValueError(f'Unsupported method: {args.method}')
        method_cls = load_method_class(args.method)
        method = method_cls(model, device, args)
        return method.run_method(data_loaders, degradation, sigma_noise)


if __name__ == '__main__':
    main()
