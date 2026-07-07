#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export RESTORA_SAVE_ALL_EVAL_IMAGES=1
export MPLBACKEND=Agg

data_root=./Data
max_batch=25
batch_size_ip=4
compute_metrics=True

### CelebA
dataset=celeba
eval_split=test
method=EGA-Flow
model_type=ot
dim_image=128
sf=2
mask_size_x=40
mask_size_y=40
p_value=0.7
sigma_y=0.01
denoise_sigma=0.2

python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}

########################################################################################################################
########################################################################################################################

### AFHQ-Cat
dataset=afhq_cat
eval_split=test
method=EGA-Flow
model_type=ot
dim_image=256
sf=4
mask_size_x=80
mask_size_y=80
p_value=0.7
sigma_y=0.01
denoise_sigma=0.2

python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
python Main.py --opts data_root ${data_root} dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} max_batch ${max_batch} batch_size_ip ${batch_size_ip} compute_metrics ${compute_metrics} dim_image ${dim_image} sf ${sf} mask_size_x ${mask_size_x} mask_size_y ${mask_size_y} p_value ${p_value} sigma_y ${sigma_y} denoise_sigma ${denoise_sigma}
