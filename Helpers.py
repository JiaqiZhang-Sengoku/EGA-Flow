import copy
import math
import os
import warnings
from ast import literal_eval
from typing import List
import numpy as np
import torch
import torchvision.transforms as v2
import yaml
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR
from Src.Models import UNet
import logging

warnings.filterwarnings("ignore", module=r"matplotlib\..*")
logging.getLogger("matplotlib.image").setLevel(logging.ERROR)


             
class CfgNode(dict):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
                                                                            
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                                         
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, super(
                CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)

    replacement_type = type(replacement)

                                                 
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))              
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'),\
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    aliases = {
        'model': 'model_type',
        'ode_steps': 'steps_ode',
        'checkpoint': 'checkpoint_name',
    }
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = aliases.get(full_key.split('.')[-1], full_key.split('.')[-1])
        if subkey in cfg:
            value = _decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(
                value, cfg[subkey], subkey, full_key
            )
            setattr(new_cfg, subkey, value)
        else:
            value = _decode_cfg_value(v)
            setattr(new_cfg, subkey, value)
    return new_cfg


            
def define_unet(args, device="cuda"):
    if args.model_type not in ["ot", "gradient_step"]:
        raise ValueError(f"Unknown model type: {args.model_type}")

    unet = UNet(
        input_channels=args.num_channels,
        input_height=args.dim_image,
        ch=32,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=6,
        attn_resolutions=(16, 8),
        resamp_with_conv=True,
    ).to(device)

                                        

    return unet


def load_model(unet, model_type, checkpoint_path, device="cuda"):
    if model_type != "ot":
        raise ValueError(f"Unsupported model type: {model_type}")

    state = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(state)
    model = unet.to(device)
    model.eval()
    return model


             
def postprocess(img, args):
    if args.dataset == "afhq_cat":
        img = (img + 1) / 2
    else:
        inv_trans = v2.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1./0.5, 1./0.5, 1./0.5])
        img = inv_trans(img)

    return img


def save_image(generated_image, output_folder, filename):
    from PIL import Image
    image_processed = generated_image.cpu().permute(0, 2, 3, 1)
    image_processed = ((image_processed + 1.0) * 127.5).clamp(0, 255)
    image_processed = image_processed.numpy().astype(np.uint8)
    image_pil = Image.fromarray(image_processed[0])
    image_pil.save(os.path.join(output_folder, filename))


def save_images(clean_img, noisy_img, rec_img, args, H_adj):
    clean_img = postprocess(clean_img.clone(), args)
    noisy_img = postprocess(noisy_img.clone(), args)
    rec_img = postprocess(rec_img.clone(), args)
    H_adj_noisy_img = postprocess(H_adj(torch.ones_like(noisy_img)), args)

    batch_size = clean_img.shape[0]
    cols = int(math.sqrt(batch_size))
    rows = int(batch_size / cols)

    clean_img = clean_img.permute(0, 2, 3, 1).cpu().data.numpy()
    noisy_img = noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    rec_img = rec_img.permute(0, 2, 3, 1).cpu().data.numpy()
    H_adj_noisy_img = H_adj_noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()

    method_name = getattr(args, 'method_config_stem', args.method)
    list_word = ['clean', 'noisy', method_name]
    for k, img in enumerate([clean_img, noisy_img, rec_img]):
        if batch_size == 1:
            fig = plt.figure()
            plt.imshow(img[0])
        elif batch_size == 2:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img[0])
            ax[1].imshow(img[1])
        else:
            fig, ax = plt.subplots(rows, cols, figsize=(20, 20))
            for i in range(rows):
                for j in range(cols):
                    if args.num_channels == 1:
                        ax[i, j].imshow(img[i + j * rows].squeeze(-1),
                                        cmap='gray', vmin=0, vmax=1)
                    else:
                        ax[i, j].imshow(img[i + j * rows])

            for ax_ in ax.flatten():
                ax_.set_xticks([])
                ax_.set_yticks([])

        plt.savefig(os.path.join(args.save_path_ip, f"batch{args.batch}_{args.problem}_{list_word[k]}.png")),
        plt.close(fig)

    if args.eval_split == 'test' or args.eval_split == 'val':
        save_all_batches = os.environ.get('RESTORA_SAVE_ALL_EVAL_IMAGES', '1') != '0'
        if args.batch < 4 or save_all_batches:
            print('Saving images one by one...')
            for i in range(batch_size):
                if args.problem == 'superresolution':
                    psnr_noisy = PSNR(clean_img[i], H_adj_noisy_img[i], data_range=1.)
                else:
                    psnr_noisy = PSNR(clean_img[i], noisy_img[i], data_range=1.)
                psnr_rec = PSNR(clean_img[i], rec_img[i], data_range=1.)

                for k, img in enumerate([clean_img, noisy_img, rec_img]):
                    fig = plt.figure()
                    plt.imshow(img[i])
                    plt.axis('off')
                    plt.title(f'PSNR={psnr_rec:4.2f}')
                    if k == 0:
                        plt.savefig(os.path.join(
                            args.save_path_ip,
                            f"im{i}_batch{args.batch}_{args.problem}_{list_word[k]}.png"),
                            bbox_inches='tight', pad_inches=0)
                    if k == 1:
                        plt.savefig(os.path.join(
                            args.save_path_ip,
                            f"im{i}_batch{args.batch}__{args.problem}_{list_word[k]}_psnr{psnr_noisy:4.2f}.png"),
                            bbox_inches='tight', pad_inches=0)
                    if k == 2:
                        plt.savefig(os.path.join(
                            args.save_path_ip,
                            f"im{i}_batch{args.batch}_{args.problem}_{list_word[k]}_psnr{psnr_rec:4.2f}.png"),
                            bbox_inches='tight', pad_inches=0)

                    plt.close(fig)
