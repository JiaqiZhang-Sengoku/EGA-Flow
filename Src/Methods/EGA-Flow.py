\
\
\
\
   

import os
import shutil
import time

import torch
from tqdm.auto import tqdm

import Helpers as helpers
from Utils import Scheduler as scheduler
from Utils.Image_Metrics import compute_psnr, compute_ssim, compute_lpips



COMMON_EGA_FLOW_DEFAULTS = {
    'steps_ode': 96,
    'correction_steps': 1,
    'guidance_enabled': True,
    'guidance_eta': 0.045,
    'guidance_beta': 0.70,
    'guidance_gamma_min': 0.30,
    'guidance_rho': 0.5,
    'guidance_max_backtrack': 5,
    'guidance_max_delta_rms': 0.075,
    'guidance_start': 0.05,
    'guidance_end': 0.995,
    'guidance_time_power': 0.65,
    'guidance_sigma_floor': 0.001,
    'guidance_stop': True,
    'guidance_stop_scale': 1.15,
    'debug_enabled': True,
    'num_restarts': 1,
    'restart_seed_base': 9173,
    'restart_reduce': 'best',
    'restart_select_metric': 'measurement_mse',
    'post_dc_enabled': False,
    'post_dc_steps': 0,
    'post_dc_lr': 0.0,
    'post_dc_max_delta_rms': 0.25,
    'post_dc_backtrack': 4,
    'post_dc_rho': 0.5,
    'final_input_blend_weight': 0.0,
    'final_data_consistency': False,
    'data_consistency_weight': 0.0,
    'momentum_enabled': False,
    'momentum_alpha': 0.4,
    'momentum_beta': 0.7,
    'momentum_min': 0.05,
    'momentum_max': 0.9,
    'noise_enabled': False,
    'noise_base_scale': 0.9,
    'noise_lambda_pos': 0.25,
    'noise_lambda_neg': 0.2,
    'noise_kappa': 0.5,
    'noise_min': 0.65,
    'noise_max': 1.15,
    'tta_hflip_enabled': False,
    'use_restora_sr_input': False,
    'clamp_output': True,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
}

EGA_FLOW_FINAL_PRESETS = {
    ('celeba', 'denoising'): {
        'steps_ode': 48,
        'final_input_blend_weight': 0.003,
        'guidance_enabled': False,
        'guidance_eta': 0.025,
        'guidance_max_delta_rms': 0.045,
        'tta_hflip_enabled': True,
    },
    ('celeba', 'box_inpainting'): {
        'steps_ode': 68,
        'guidance_enabled': False,
        'guidance_eta': 0.055,
        'num_restarts': 3,
        'restart_reduce': 'mean',
        'restart_select_metric': 'measurement_plus_range',
        'tta_hflip_enabled': True,
    },
    ('celeba', 'superresolution'): {
        'steps_ode': 176,
        'momentum_enabled': True,
        'noise_enabled': True,
        'noise_lambda_neg': 0.12,
    },
    ('celeba', 'random_inpainting'): {
        'steps_ode': 208,
        'correction_steps': 2,
        'momentum_enabled': True,
        'noise_enabled': True,
        'noise_base_scale': 0.84,
        'noise_lambda_pos': 0.5,
        'noise_lambda_neg': 0.06,
    },
    ('afhq_cat', 'denoising'): {
        'steps_ode': 64,
        'guidance_enabled': False,
    },
    ('afhq_cat', 'box_inpainting'): {
        'mask_size_x': 80,
        'mask_size_y': 80,
        'steps_ode': 56,
        'guidance_enabled': False,
        'guidance_eta': 0.055,
        'num_restarts': 3,
        'restart_reduce': 'mean',
        'restart_select_metric': 'measurement_plus_range',
        'tta_hflip_enabled': True,
    },
    ('afhq_cat', 'superresolution'): {
        'sf': 4,
        'steps_ode': 192,
        'momentum_enabled': True,
        'noise_enabled': True,
        'noise_base_scale': 0.88,
        'noise_lambda_pos': 0.3,
        'noise_lambda_neg': 0.1,
    },
    ('afhq_cat', 'random_inpainting'): {
        'steps_ode': 160,
        'correction_steps': 2,
        'guidance_enabled': False,
        'num_restarts': 3,
        'restart_reduce': 'mean',
        'restart_select_metric': 'measurement_plus_range',
        'tta_hflip_enabled': True,
    },
}


def apply_ega_flow_final_preset(args):
                                                           
    for name, value in COMMON_EGA_FLOW_DEFAULTS.items():
        setattr(args, name, value)
    key = (str(args.dataset), str(args.problem))
    for name, value in EGA_FLOW_FINAL_PRESETS.get(key, {}).items():
        setattr(args, name, value)
    args.ega_flow_fixed_preset = True
    return args


def batch_rms(x, eps=1e-8):
    dims = tuple(range(1, x.ndim))
    return torch.sqrt(torch.mean(x.float() ** 2, dim=dims, keepdim=True) + eps)


def normalize_rms(x, eps=1e-8):
    return x / batch_rms(x, eps=eps)


def batch_cosine(a, b, eps=1e-8):
    dims = tuple(range(1, a.ndim))
    dot = torch.sum(a.float() * b.float(), dim=dims, keepdim=True)
    a_norm = torch.sqrt(torch.sum(a.float() ** 2, dim=dims, keepdim=True) + eps)
    b_norm = torch.sqrt(torch.sum(b.float() ** 2, dim=dims, keepdim=True) + eps)
    return dot / (a_norm * b_norm + eps)


def time_weight(t, guide_start=0.05, guide_end=0.995, power=0.5, device=None, dtype=None):
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=device, dtype=dtype or torch.float32)
    s = (t - guide_start) / max(guide_end - guide_start, 1e-8)
    return torch.clamp(s, 0.0, 1.0) ** power


class EndpointOperator:
    def __init__(self, H, H_adj):
        self.H = H
        self.H_adj = H_adj

    def loss(self, x, y, sigma_y=0.01, sigma_floor=1e-3):
        denom = max(float(sigma_y) ** 2, float(sigma_floor) ** 2)
        err = self.H(x) - y
        dims = tuple(range(1, err.ndim))
        return 0.5 * torch.mean(err.float() ** 2, dim=dims) / denom

    def grad(self, x, y, sigma_y=0.01, sigma_floor=1e-3):
        denom = max(float(sigma_y) ** 2, float(sigma_floor) ** 2)
        return self.H_adj(self.H(x) - y) / denom


class SamplerState:
    def __init__(self):
        self.prev_g_unit = None
        self.prev_m = None
        self.prev_alignment = None

    def reset(self):
        self.prev_g_unit = None
        self.prev_m = None
        self.prev_alignment = None


class FlippedDegradation:
    def __init__(self, degradation, dims=(-1,)):
        self.degradation = degradation
        self.dims = dims

    def H(self, x):
        return torch.flip(self.degradation.H(torch.flip(x, dims=self.dims)), dims=self.dims)

    def H_adj(self, y):
        return torch.flip(self.degradation.H_adj(torch.flip(y, dims=self.dims)), dims=self.dims)


class EGAFlow(object):
    def __init__(self, model, device, args):
        if str(getattr(args, 'dataset', '')) in {'celeba', 'afhq_cat'}:
            apply_ega_flow_final_preset(args)
        else:
            args.ega_flow_fixed_preset = False
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.debug_history = []

    def model_forward(self, x, t):
        return self.model(x, t)

    def _cfg(self, name, default):
        return getattr(self.args, name, default)

    def _task_cfg(self, stem, default):
        task_name = f'{self.args.problem}_{stem}'
        return self._cfg(task_name, self._cfg(stem, default))

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def _guidance_enabled(self):
        return self._as_bool(self._cfg('guidance_enabled', True)) and float(self._task_cfg('guidance_eta', 0.0)) > 0.0

    def _hflip_tta_enabled(self):
        return self._as_bool(self._task_cfg('tta_hflip_enabled', self._cfg('tta_hflip_enabled', False)))

    def _clamp_range(self):
        if not self._as_bool(self._cfg('clamp_output', False)):
            return None
        lo = float(self._cfg('clamp_min', -1.0))
        hi = float(self._cfg('clamp_max', 1.0))
        return lo, hi

    def _maybe_clamp(self, x):
        clamp_range = self._clamp_range()
        if clamp_range is None:
            return x
        return x.clamp(clamp_range[0], clamp_range[1])

    def _final_data_consistency(self, x, input_img, mask):
        if not self._as_bool(self._cfg('final_data_consistency', False)):
            return x
        task_weight_name = f'{self.args.problem}_data_consistency_weight'
        weight = float(self._cfg(task_weight_name, self._cfg('data_consistency_weight', 1.0)))
        weight = max(0.0, min(1.0, weight))
        if weight == 0.0:
            return x
        known = weight * input_img + (1.0 - weight) * x
        return mask * known + (1 - mask) * x

    def _final_input_blend(self, x, input_img):
        weight = float(self._task_cfg('final_input_blend_weight', self._cfg('final_input_blend_weight', 0.0)))
        weight = max(0.0, min(1.0, weight))
        if weight == 0.0:
            return x
        return (1.0 - weight) * x + weight * input_img

    def _post_operator_consistency(self, x, y, operator):
        if operator is None or not self._as_bool(self._task_cfg('post_dc_enabled', self._cfg('post_dc_enabled', False))):
            return x

        steps = int(self._task_cfg('post_dc_steps', self._cfg('post_dc_steps', 0)))
        lr = float(self._task_cfg('post_dc_lr', self._cfg('post_dc_lr', 0.0)))
        if steps <= 0 or lr <= 0:
            return x

        max_delta_rms = float(self._task_cfg('post_dc_max_delta_rms', self._cfg('post_dc_max_delta_rms', 0.25)))
        backtracks = int(self._task_cfg('post_dc_backtrack', self._cfg('post_dc_backtrack', 4)))
        rho = float(self._task_cfg('post_dc_rho', self._cfg('post_dc_rho', 0.5)))
        clamp_range = self._clamp_range()

        def loss_by_batch(z):
            err = operator.H(z) - y
            return torch.mean(err.float() ** 2, dim=tuple(range(1, err.ndim)), keepdim=True)

        with torch.no_grad():
            out = x
            for _ in range(steps):
                loss_old = loss_by_batch(out)
                grad = operator.H_adj(operator.H(out) - y)
                step = -lr * grad
                delta_rms = batch_rms(step)
                delta_scale = torch.clamp(max_delta_rms / (delta_rms + 1e-8), max=1.0)
                step = step * delta_scale
                scale = 1.0
                accepted = torch.zeros(out.shape[0], device=out.device, dtype=torch.bool)
                best = out
                for _bt in range(backtracks + 1):
                    trial = out + scale * step
                    if clamp_range is not None:
                        trial = trial.clamp(clamp_range[0], clamp_range[1])
                    loss_new = loss_by_batch(trial)
                    ok = (loss_new <= loss_old + 1e-12).view(-1)
                    accept_now = (~accepted) & ok
                    best = torch.where(accept_now.view(-1, 1, 1, 1), trial, best)
                    accepted = accepted | ok
                    if bool(accepted.all().item()):
                        break
                    scale *= rho
                out = torch.where(accepted.view(-1, 1, 1, 1), best, out)
            return out.detach()

    def _restart_count(self):
        return max(1, int(self._task_cfg('num_restarts', self._cfg('num_restarts', 1))))

    def _restart_seed(self, batch, restart):
        base = int(self._task_cfg('restart_seed_base', self._cfg('restart_seed_base', 9173)))
        return base + int(batch) * 1009 + int(restart)

    def _selection_score(self, output, measurement, operator):
        metric = str(self._task_cfg('restart_select_metric', self._cfg('restart_select_metric', 'measurement_mse')))
        if metric == 'range_penalty' or operator is None:
            over = torch.relu(output - 1.0) + torch.relu(-1.0 - output)
            return torch.mean(over.float() ** 2, dim=tuple(range(1, over.ndim)))
        err = operator.H(output) - measurement
        score = torch.mean(err.float() ** 2, dim=tuple(range(1, err.ndim)))
        if metric == 'measurement_plus_range':
            over = torch.relu(output - 1.0) + torch.relu(-1.0 - output)
            score = score + 0.01 * torch.mean(over.float() ** 2, dim=tuple(range(1, over.ndim)))
        return score

    def _combine_restarts(self, outputs, measurement, operator):
        if len(outputs) == 1:
            return outputs[0]

        reduce = str(self._task_cfg('restart_reduce', self._cfg('restart_reduce', 'best')))
        stacked = torch.stack(outputs, dim=0)
        if reduce == 'mean':
            return stacked.mean(dim=0)
        if reduce == 'median':
            return stacked.median(dim=0).values

        scores = torch.stack([self._selection_score(out, measurement, operator) for out in outputs], dim=0)
        idx = torch.argmin(scores, dim=0)
        gathered = []
        for b in range(stacked.shape[1]):
            gathered.append(stacked[idx[b], b])
        return torch.stack(gathered, dim=0)

    def _adaptive_noise_scale(self, alignment, t):
        base = float(self._task_cfg('noise_base_scale', self._cfg('noise_base_scale', 0.9)))
        lambda_pos = float(self._task_cfg('noise_lambda_pos', self._cfg('noise_lambda_pos', 0.25)))
        lambda_neg = float(self._task_cfg('noise_lambda_neg', self._cfg('noise_lambda_neg', 0.2)))
        kappa = float(self._task_cfg('noise_kappa', self._cfg('noise_kappa', 0.5)))
        lo = float(self._task_cfg('noise_min', self._cfg('noise_min', 0.65)))
        hi = float(self._task_cfg('noise_max', self._cfg('noise_max', 1.15)))
        if not torch.is_tensor(alignment):
            alignment = torch.tensor(float(alignment), device=t.device, dtype=t.dtype)
        alignment = alignment.to(device=t.device, dtype=t.dtype).clamp(-1.0, 1.0)
        while alignment.ndim < t.ndim:
            alignment = alignment.view(*alignment.shape, *([1] * (t.ndim - alignment.ndim)))
        tau = 1.0 - kappa * t
        raw = torch.where(alignment > 0.0, base + lambda_pos * alignment, base - lambda_neg * tau * alignment.square())
        return torch.clamp(raw, lo, hi)

    def _append_debug(self, debug):
        if self._as_bool(self._cfg('debug_enabled', False)):
            self.debug_history.append(debug)

    def _debug_summary(self):
        if not self.debug_history:
            return {}
        keys = sorted(self.debug_history[0].keys())
        summary = {}
        for key in keys:
            vals = [item[key] for item in self.debug_history if key in item]
            if vals and isinstance(vals[0], (float, int)):
                summary[key] = sum(vals) / len(vals)
        return summary

    def _guided_step(self, x, t, dt, y, operator, state):
        device = x.device
        dtype = x.dtype
        t_float = float(t)
        dt_float = float(dt)
        t_next = min(max(t_float + dt_float, 0.0), 1.0)
        batch_size = x.shape[0]

        eta0 = float(self._task_cfg('guidance_eta', 0.04))
        beta = float(self._task_cfg('guidance_beta', 0.70))
        gamma_min = float(self._task_cfg('guidance_gamma_min', 0.25))
        rho = float(self._task_cfg('guidance_rho', 0.5))
        max_backtrack = int(self._task_cfg('guidance_max_backtrack', 4))
        max_delta_rms = float(self._task_cfg('guidance_max_delta_rms', 0.08))
        guide_start = float(self._task_cfg('guidance_start', 0.05))
        guide_end = float(self._task_cfg('guidance_end', 0.995))
        time_power = float(self._task_cfg('guidance_time_power', 0.5))
        sigma_floor = float(self._task_cfg('guidance_sigma_floor', 1e-3))
        noise_stop = self._as_bool(self._task_cfg('guidance_stop', True))
        noise_stop_scale = float(self._task_cfg('guidance_stop_scale', 1.2))
        adaptive_momentum = self._as_bool(self._task_cfg('momentum_enabled', self._cfg('momentum_enabled', False)))
        momentum_alpha = float(self._task_cfg('momentum_alpha', self._cfg('momentum_alpha', 0.4)))
        momentum_beta = float(self._task_cfg('momentum_beta', self._cfg('momentum_beta', 0.7)))
        momentum_min = float(self._task_cfg('momentum_min', self._cfg('momentum_min', 0.05)))
        momentum_max = float(self._task_cfg('momentum_max', self._cfg('momentum_max', 0.9)))
        clamp_range = self._clamp_range()

        with torch.no_grad():
            t_batch = torch.full((batch_size,), t_float, device=device, dtype=dtype)
            v = self.model_forward(x, t_batch)
            x_prior = x + dt_float * v

            if t_next >= 1.0 - 1e-6:
                v_prior = torch.zeros_like(x_prior)
                e = x_prior
            else:
                t_next_batch = torch.full((batch_size,), t_next, device=device, dtype=dtype)
                v_prior = self.model_forward(x_prior, t_next_batch)
                e = x_prior + (1.0 - t_next) * v_prior

            g = operator.grad(e, y, sigma_y=self.args.sigma_noise, sigma_floor=sigma_floor)
            loss_old = operator.loss(e, y, sigma_y=self.args.sigma_noise, sigma_floor=sigma_floor)
            meas_err = operator.H(e) - y
            meas_mse = torch.mean(meas_err.float() ** 2, dim=tuple(range(1, meas_err.ndim)))

            g_unit = torch.nan_to_num(normalize_rms(g), nan=0.0, posinf=0.0, neginf=0.0)
            prior_dir = e - x_prior
            if bool((batch_rms(prior_dir).mean() < 1e-8).item()):
                prior_dir = v_prior
            prior_unit = torch.nan_to_num(normalize_rms(prior_dir), nan=0.0, posinf=0.0, neginf=0.0)

            cos_prior = batch_cosine(-g_unit, prior_unit).clamp(-1.0, 1.0)
            s_prior = 0.5 * (1.0 + cos_prior)
            gamma_prior = gamma_min + (1.0 - gamma_min) * s_prior

            if state.prev_g_unit is None:
                cos_temp = torch.ones_like(cos_prior)
                gamma_temp = torch.ones_like(gamma_prior)
            else:
                cos_temp = batch_cosine(g_unit, state.prev_g_unit).clamp(-1.0, 1.0)
                s_temp = 0.5 * (1.0 + cos_temp)
                gamma_temp = gamma_min + (1.0 - gamma_min) * s_temp

            if state.prev_m is None:
                m = g_unit
                beta_eff = torch.zeros_like(gamma_prior)
            else:
                if adaptive_momentum:
                    beta_eff = torch.clamp(momentum_alpha + momentum_beta * cos_prior, momentum_min, momentum_max)
                else:
                    cos_m = batch_cosine(g_unit, state.prev_m).clamp(-1.0, 1.0)
                    beta_eff = beta * 0.5 * (1.0 + cos_m)
                m = beta_eff * state.prev_m + (1.0 - beta_eff) * g_unit
                m = torch.nan_to_num(normalize_rms(m), nan=0.0, posinf=0.0, neginf=0.0)

            gamma = gamma_prior * gamma_temp
            w_t = time_weight(
                t_next,
                guide_start=guide_start,
                guide_end=guide_end,
                power=time_power,
                device=device,
                dtype=dtype,
            ).view(1, 1, 1, 1)
            eta_eff = eta0 * w_t * gamma

            if noise_stop and float(self.args.sigma_noise) > 0:
                stop = meas_mse <= (noise_stop_scale * float(self.args.sigma_noise)) ** 2
                eta_eff = torch.where(stop.view(-1, 1, 1, 1), torch.zeros_like(eta_eff), eta_eff)

            accepted = torch.zeros(batch_size, device=device, dtype=torch.bool)
            n_backtrack = torch.zeros(batch_size, device=device, dtype=torch.long)
            e_guided = e
            step_scale = eta_eff

            for k in range(max_backtrack + 1):
                delta = -step_scale * m
                delta_rms = batch_rms(delta)
                delta_scale = torch.clamp(max_delta_rms / (delta_rms + 1e-8), max=1.0)
                delta = delta * delta_scale
                e_trial = e + delta
                if clamp_range is not None:
                    e_trial = e_trial.clamp(clamp_range[0], clamp_range[1])

                loss_new = operator.loss(e_trial, y, sigma_y=self.args.sigma_noise, sigma_floor=sigma_floor)
                ok = loss_new <= loss_old + 1e-12
                accept_now = (~accepted) & ok
                e_guided = torch.where(accept_now.view(-1, 1, 1, 1), e_trial, e_guided)
                n_backtrack = torch.where(accept_now, torch.full_like(n_backtrack, k), n_backtrack)
                accepted = accepted | ok
                if bool(accepted.all().item()):
                    break
                step_scale = step_scale * rho

            e_guided = torch.where(accepted.view(-1, 1, 1, 1), e_guided, e)

            if t_next >= 1.0 - 1e-6:
                x_next = e_guided
            else:
                denom = max(1.0 - t_next, 1e-6)
                x0_hat = (x_prior - t_next * e) / denom
                x_next = t_next * e_guided + (1.0 - t_next) * x0_hat

            if clamp_range is not None:
                x_next = x_next.clamp(clamp_range[0], clamp_range[1])

            state.prev_g_unit = g_unit.detach()
            state.prev_m = m.detach()
            state.prev_alignment = cos_prior.detach()

            debug = {
                'loss_old_mean': float(loss_old.mean().item()),
                'loss_new_mean': float(operator.loss(e_guided, y, sigma_y=self.args.sigma_noise, sigma_floor=sigma_floor).mean().item()),
                'meas_mse_mean': float(meas_mse.mean().item()),
                'cos_prior_mean': float(cos_prior.mean().item()),
                'cos_temp_mean': float(cos_temp.mean().item()),
                'momentum_enabled': int(adaptive_momentum),
                'beta_eff_mean': float(beta_eff.mean().item()),
                'gamma_mean': float(gamma.mean().item()),
                'eta_eff_mean': float(eta_eff.mean().item()),
                'accepted_ratio': float(accepted.float().mean().item()),
                'backtrack_mean': float(n_backtrack.float().mean().item()),
                'grad_rms_mean': float(batch_rms(g).mean().item()),
                'delta_rms_mean': float(batch_rms(e_guided - e).mean().item()),
            }

        self._append_debug(debug)
        return x_next.detach()

    def sample_denoising(self, input_img, degradation=None):
        steps_ode = self.args.steps_ode
        device = input_img.device

        x = torch.randn_like(input_img, device=device)
        x_obs = input_img * (1 - self.args.sigma_noise)
        mask = torch.ones(input_img.shape, device=device)
        operator = EndpointOperator(degradation.H, degradation.H_adj) if degradation is not None else None
        state = SamplerState()

        torch_linspace = torch.linspace(0, 1, int(steps_ode), device=device)
        delta_t = 1 / len(torch_linspace)

        for t in torch_linspace:
            t_float = float(t.item())
            if t_float < (1 - self.args.sigma_noise):
                x = mask * x_obs + (1 - mask) * x
                state.reset()
            else:
                if self._guidance_enabled() and operator is not None:
                    x = self._guided_step(x, t_float, delta_t, input_img, operator, state)
                else:
                    x = x + delta_t * self.model(x, torch.tensor(t_float, device=device).repeat(x.shape[0]))

        x = self._final_input_blend(x, input_img)
        return self._maybe_clamp(x)

    def sample_mask_based(
        self,
        input_img,
        mask,
        measurement=None,
        degradation=None,
        sample_id=0,
        progress=False,
        debug=False,
    ):
        batch_size = input_img.shape[0]
        output_folder = self.args.save_path_ip

        if debug:
            helpers.save_image(input_img, output_folder, 'input_img.png')
            helpers.save_image(mask, output_folder, 'mask.png')

        x = torch.randn_like(input_img, device=self.device)
        pred_x_start = None
        out_sample = x.clone()
        steps_ode = self.args.steps_ode
        correction_steps = self.args.correction_steps
        operator = EndpointOperator(degradation.H, degradation.H_adj) if degradation is not None else None
        y = measurement if measurement is not None else input_img
        state = SamplerState()

        if correction_steps < 1:
            raise ValueError("Number of correction steps must be >= 1.")

        times = scheduler.get_schedule_jump(
            t_T=steps_ode,
            n_sample=1,
            jump_length=1,
            jump_n_sample=correction_steps + 1,
        )

        times = [((x - min(times)) / (max(times) - min(times))) for x in times]
        times.reverse()
        time_pairs = list(zip(times[:-1], times[1:]))

        if progress:
            time_pairs = tqdm(time_pairs)

        for t_last, t_cur in time_pairs:
            if debug:
                print("t_last, t_cur: ", t_last, t_cur)

            t_last_t = torch.tensor([t_last] * batch_size, device=self.device).view(batch_size, 1, 1, 1)
            t_cur_t = torch.tensor([t_cur] * batch_size, device=self.device).view(batch_size, 1, 1, 1)

            if t_last < t_cur:
                with torch.no_grad():
                    if pred_x_start is not None:
                        eps = torch.randn_like(x)
                        z_prim = t_last_t * input_img + (1 - t_last_t) * eps
                        x = mask * z_prim + (1 - mask) * x

                        if debug:
                            helpers.save_image(mask * z_prim, output_folder, f'{sample_id}_known.png')
                            helpers.save_image((1 - mask) * x, output_folder, f'{sample_id}_unknown.png')

                    delta_t = float(t_cur - t_last)
                    if self._guidance_enabled() and operator is not None:
                        x = self._guided_step(x, float(t_last), delta_t, y, operator, state)
                    else:
                        x = x + delta_t * self.model(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                    out_sample = x.clone()

                    if debug:
                        helpers.save_image(out_sample, output_folder, 'out_sample.png')

                    pred_x_start = True
            else:
                adaptive_noise_scale = 1.0
                if self._as_bool(self._task_cfg('noise_enabled', self._cfg('noise_enabled', False))) and state.prev_alignment is not None:
                    adaptive_noise_scale = self._adaptive_noise_scale(state.prev_alignment, t_cur_t)
                state.reset()
                x_1_prim = x + (1 - t_last_t) * self.model(x, torch.tensor(t_last, device=self.device).repeat(batch_size))
                x = t_cur_t * x_1_prim + (1 - t_cur_t) * adaptive_noise_scale * torch.randn_like(x)

        out_sample = self._final_data_consistency(out_sample, input_img, mask)
        return self._maybe_clamp(out_sample)

    def _sample_problem_once(self, noisy_img, clean_img, degradation, mask):
        if self.args.problem == 'denoising':
            return self.sample_denoising(input_img=noisy_img, degradation=degradation)
        if self.args.problem == 'superresolution':
            superresolution_input = degradation.H_adj(noisy_img).to(self.device)
            if self._as_bool(self._cfg('use_restora_sr_input', False)):
                clean_device = clean_img.to(self.device)
                superresolution_input = clean_device * mask + torch.randn_like(clean_device) * self.args.sigma_noise
            return self.sample_mask_based(
                input_img=superresolution_input,
                mask=mask,
                measurement=noisy_img,
                degradation=degradation,
            )
        return self.sample_mask_based(
            input_img=noisy_img,
            mask=mask,
            measurement=noisy_img,
            degradation=degradation,
        )

    def _maybe_hflip_tta(self, output, noisy_img, clean_img, degradation):
        if not self._hflip_tta_enabled():
            return output
        flipped_degradation = FlippedDegradation(degradation, dims=(-1,))
        noisy_flip = torch.flip(noisy_img, dims=(-1,))
        clean_flip = torch.flip(clean_img, dims=(-1,))
        mask_flip = flipped_degradation.H_adj(torch.ones_like(noisy_flip)).to(self.device)
        output_flip = self._sample_problem_once(noisy_flip, clean_flip, flipped_degradation, mask_flip)
        return 0.5 * (output + torch.flip(output_flip, dims=(-1,)))

    def solve_ip(self, test_loader, degradation):
        H, H_adj = degradation.H, degradation.H_adj

        loader = iter(test_loader)
        psnrs, ssims, lpips = [], [], []

        for batch in range(self.args.max_batch):
            self.args.batch = batch

            (clean_img, labels) = next(loader)
            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * self.args.sigma_noise
            noisy_img, clean_img = noisy_img.to(self.device), clean_img.to('cpu')

            mask = H_adj(torch.ones_like(noisy_img)).to(self.device)

            operator = EndpointOperator(H, H_adj)
            outputs = []
            restart_count = self._restart_count()
            with torch.no_grad():
                for restart in range(restart_count):
                    if restart_count > 1:
                        torch.manual_seed(self._restart_seed(batch, restart))
                    output = self._sample_problem_once(noisy_img, clean_img, degradation, mask)
                    output = self._maybe_hflip_tta(output, noisy_img, clean_img, degradation)
                    output = self._post_operator_consistency(output, noisy_img, operator)
                    outputs.append(output.detach())
                output = self._combine_restarts(outputs, noisy_img, operator)
                output = self._maybe_clamp(output)

            restored_img = output.detach().clone()

            if self.args.compute_metrics:
                psnr_rec, psnr_noisy = compute_psnr(clean_img, noisy_img, restored_img, self.args, H_adj)
                print(f"Batch {batch}: psnr_rec={psnr_rec}, psnr_noisy={psnr_noisy}")
                psnrs.append(psnr_rec)
                ssim_rec, ssim_noisy = compute_ssim(clean_img, noisy_img, restored_img, self.args, H_adj)
                print(f"Batch {batch}: ssim_rec={ssim_rec}, ssim_noisy={ssim_noisy}")
                ssims.append(ssim_rec)
                lpip_rec, lpip_noisy = compute_lpips(clean_img, noisy_img, restored_img, self.args, H_adj)
                lpips.append(lpip_rec)

            if self._as_bool(self._cfg('debug_enabled', False)):
                summary = self._debug_summary()
                if summary:
                    print("Debug:", ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))

            helpers.save_images(clean_img, noisy_img, restored_img, self.args, H_adj)

        return psnrs, ssims, lpips

    def _params_string(self):
        names = [
            'steps_ode',
            'correction_steps',
            'guidance_enabled',
            'guidance_eta',
            'guidance_beta',
            'guidance_gamma_min',
            'guidance_max_delta_rms',
            'guidance_max_backtrack',
            'guidance_start',
            'guidance_end',
            'guidance_time_power',
            'guidance_stop',
            'guidance_stop_scale',
            'momentum_enabled',
            'momentum_alpha',
            'momentum_beta',
            'momentum_min',
            'momentum_max',
            'noise_enabled',
            'noise_base_scale',
            'noise_lambda_pos',
            'noise_lambda_neg',
            'num_restarts',
            'restart_reduce',
            'restart_select_metric',
            'tta_hflip_enabled',
            'post_dc_enabled',
            'post_dc_steps',
            'post_dc_lr',
            'post_dc_max_delta_rms',
            'final_input_blend_weight',
            'final_data_consistency',
            'data_consistency_weight',
            'clamp_output',
        ]
        return ', '.join(f'{name}={self._task_cfg(name, "NA")}' for name in names)

    def run_method(self, data_loaders, degradation, sigma_noise):
        print(f'Params: {self._params_string()}\n')

        self.args.sigma_noise = sigma_noise
        self.debug_history = []

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
                    f'Params: {self._params_string()}\n'
                    f'---------------------------------------------------------\n'
                )

                for idx, (psnr, ssim, lpip) in enumerate(zip(psnrs, ssims, lpips)):
                    file.write(f'Batch {idx}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, LPIPS = {lpip:.4f}\n')

                file.write(f'---------------------------------------------------------\n')
                file.write(f'Average PSNR = {avg_psnr:.4f}\n')
                file.write(f'Average SSIM = {avg_ssim:.4f}\n')
                file.write(f'Average LPIPS = {avg_lpips:.4f}\n')
                file.write(f'Total time = {total_time}\n')

                summary = self._debug_summary()
                if summary:
                    file.write('Debug summary:\n')
                    for key, value in summary.items():
                        file.write(f'{key} = {value:.6f}\n')
