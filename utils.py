import torch
import pyro.distributions as dist

from py_replay_bg.model.t1d_model_single_meal import T1DModelSingleMeal
from py_replay_bg.environment import Environment
from py_replay_bg.data import ReplayBGData

import pandas as pd 
import numpy as np 
from scipy.stats import wasserstein_distance

N_PARAMS=9
N_PARAMS_HAT=18
class CustomPrior:
    def __init__(self, model, rbg_data, device, fixed_beta=False):
        self.device = device
        self.model = model
        self.rbg_data = rbg_data
        self.n_params = N_PARAMS
        self.fixed_beta = fixed_beta
        if fixed_beta:
            self.n_params -= 1
            
        self.VG = torch.tensor(model.model_parameters.VG, device=self.device)

        # Distributions
        self.gamma_SI = dist.Gamma(torch.tensor(3.3, device=self.device), torch.tensor(1 / 5e-4, device=self.device))
        self.normal_p2 = dist.Normal(torch.tensor(0.11, device=self.device), torch.tensor(0.004, device=self.device))
        self.normal_Gb = dist.Normal(torch.tensor(119.13, device=self.device), torch.tensor(7.11, device=self.device))
        self.Gb_low = torch.tensor(70.0, device=self.device)
        self.Gb_high = torch.tensor(180.0, device=self.device)

        self.ln_SG = dist.Normal(torch.tensor(-3.8, device=self.device), torch.tensor(0.5, device=self.device))
        self.ln_ka2 = dist.Normal(torch.tensor(-4.2875, device=self.device), torch.tensor(0.4274, device=self.device))
        self.ln_kd = dist.Normal(torch.tensor(-3.5090, device=self.device), torch.tensor(0.6187, device=self.device))
        self.ln_kempt = dist.Normal(torch.tensor(-1.9646, device=self.device), torch.tensor(0.7069, device=self.device))
        self.ln_kabs = dist.Normal(torch.tensor(-5.4591, device=self.device), torch.tensor(1.4396, device=self.device))

    def truncated_normal(self, mu, sigma, low, high, size):
        samples = torch.normal(mu, sigma, size=size, device=self.device)
        mask = (samples > low) & (samples < high)
        while not mask.all():
            resample = torch.normal(mu, sigma, size=((~mask).sum().item(),), device=self.device)
            samples[~mask] = resample
            mask = (samples > low) & (samples < high)
        return samples

    def truncated_lognormal(self, mu, sigma, low, high, size):
        samples = torch.exp(torch.normal(mu, sigma, size=size, device=self.device))
        mask = (samples >= low) & (samples <= high)
        while not mask.all():
            resample = torch.exp(torch.normal(mu, sigma, size=((~mask).sum().item(),), device=self.device))
            samples[~mask] = resample
            mask = (samples > low) & (samples < high)
        return samples

    def sample(self, shape=torch.Size()):
        if len(shape) == 0:
            n_samples = 1
            squeeze = True
        else:
            n_samples = shape[0]
            squeeze = False

        SI = self.gamma_SI.sample((n_samples,)) / self.VG
        p2 = torch.normal(0.11, 0.004, size=(n_samples,), device=self.device).pow(2)
        Gb = self.truncated_normal(119.13, 7.11, low=70, high=180, size=(n_samples,))
        SG = self.truncated_lognormal(-3.8, 0.5, low=1e-6, high=1.0, size=(n_samples,))
        ka2 = self.truncated_lognormal(-4.2875, 0.4274, low=1e-5, high=1.0, size=(n_samples,))
        kd = self.truncated_lognormal(-3.5090, 0.6187, low=ka2.min().item(), high=1.0, size=(n_samples,))

        mask_ka2 = ka2 < kd
        while not mask_ka2.all():
            ka2[~mask_ka2] = self.truncated_lognormal(-4.2875, 0.4274, 1e-5, 1.0, size=((~mask_ka2).sum().item(),))
            kd[~mask_ka2] = self.truncated_lognormal(-3.5090, 0.6187, ka2[~mask_ka2].min().item(), 1.0, size=((~mask_ka2).sum().item(),))
            mask_ka2 = ka2 < kd

        kempt = self.truncated_lognormal(-1.9646, 0.7069, low=1e-5, high=1.0, size=(n_samples,))
        kabs = self.truncated_lognormal(-5.4591, 1.4396, low=1e-6, high=kempt.min().item(), size=(n_samples,))
        
        if self.fixed_beta:
            theta = torch.stack([Gb, SG, p2, ka2, kd, kempt, SI, kabs], dim=1)
        else:
            beta = torch.rand(n_samples, device=self.device) * 60
            theta = torch.stack([Gb, SG, p2, ka2, kd, kempt, SI, kabs, beta], dim=1)

        # Sample x0s
        x0s = np.array([self.model.sample_x0(self.rbg_data, t.cpu().numpy()) for t in theta])
        x0s = torch.tensor(x0s, dtype=torch.float32, device=self.device)

        samples = torch.cat([theta, x0s], dim=1)
        if squeeze:
            return samples.squeeze(0)
        return samples

    def log_prob(self, samples):
        theta = samples[:, :self.n_params]
        x0 = samples[:, self.n_params:]

        # If any x0 values are negative, assign -inf
        mask_invalid = (x0 < 0).any(dim=1)
        if self.fixed_beta:
            Gb, SG, p2, ka2, kd, kempt, SI, kabs = theta.unbind(dim=1)
        else:
            Gb, SG, p2, ka2, kd, kempt, SI, kabs, beta = theta.unbind(dim=1)

        log_prob = torch.zeros(samples.shape[0], device=self.device)

        log_prob += self.gamma_SI.log_prob(SI * self.VG) + torch.log(self.VG)
        p2_sqrt = torch.sqrt(p2)
        log_prob += self.normal_p2.log_prob(p2_sqrt) - torch.log(2 * p2_sqrt)
        log_prob += self.normal_Gb.log_prob(Gb) - torch.log(self.normal_Gb.cdf(self.Gb_high) - self.normal_Gb.cdf(self.Gb_low))
        log_prob += self.ln_SG.log_prob(torch.log(SG)) - torch.log(SG)
        log_prob += self.ln_ka2.log_prob(torch.log(ka2)) - torch.log(ka2)
        log_prob += self.ln_kd.log_prob(torch.log(kd)) - torch.log(kd)
        log_prob += self.ln_kempt.log_prob(torch.log(kempt)) - torch.log(kempt)
        log_prob += self.ln_kabs.log_prob(torch.log(kabs)) - torch.log(kabs)
        log_prob += -torch.log(torch.tensor(60.0, device=self.device))

        # Constraint: ka2 < kd, kabs < kempt
        mask_invalid |= (ka2 >= kd) | (kabs >= kempt)
        
        log_prob[mask_invalid] = -float('inf')

        return log_prob
    
    
def get_prior(model: T1DModelSingleMeal, rbg_data: ReplayBGData, device = torch.device('cpu'), fixed_beta=False) -> CustomPrior:
    custom_prior = CustomPrior(model=model, rbg_data=rbg_data, device=device, fixed_beta=fixed_beta)
    return custom_prior

def simulate_one(args):
    theta_np, model, rbg_data = args
    try:
        if len(theta_np) > N_PARAMS:
            x0 = theta_np[-N_PARAMS:]
            theta = theta_np[:-N_PARAMS]
        else:
            x0 = None
            theta = theta_np
        all_states, cgm = model.sbi_simulate(rbg_data, x0, theta)
    except Exception as e:
        print(e)
        all_states, cgm = None, None
        
    return all_states, cgm

def get_model_and_rbg_data(data_path, patient_info_path, glucose_sequence=None, cho=None, fixed_beta=False):
    data = pd.read_csv(data_path)
    data.t = pd.to_datetime(data['t'])
    if glucose_sequence is not None:
        common_len = min(len(data['glucose']), len(glucose_sequence))
        data['glucose'].values[:common_len] = glucose_sequence[:common_len]
    if cho is not None:
        data.cho = cho
    patient_info = pd.read_csv(patient_info_path)
    p = np.where(patient_info['patient'] == 1)[0][0]
    bw = float(patient_info.bw.values[p])
    u2ss = float(patient_info.u2ss.values[p])

    # Environment and model setup
    env = Environment(save_name='ori', save_folder='./')
    model = T1DModelSingleMeal(data=data, bw=bw, u2ss=u2ss, environment=env, fixed_beta=fixed_beta)
    rbg_data = ReplayBGData(data=data, model=model, environment=env)
    return model, rbg_data


def print_summary(name, mard_med, rmsd_med):
    print(f"{name} MARD: {np.mean(mard_med) * 100:.02f} ± {np.std(mard_med)*100:.02f}%, "
          f"RMSD: {np.mean(rmsd_med):.02f} ± {np.std(rmsd_med):.02f}")

def compute_cgm_metrics(pred_median, true_x):
    mard = np.mean(np.abs((pred_median - true_x) / true_x))
    rmsd = np.sqrt(np.mean((pred_median - true_x) ** 2))
    return mard, rmsd

def coverage(samples, true_val, lower=2.5, upper=97.5):
    """Check if true_val lies within the credible interval."""
    lower_bound = np.percentile(samples, lower)
    upper_bound = np.percentile(samples, upper)
    return lower_bound <= true_val <= upper_bound

def compute_params_metrics(samples, true_val):
    """
    Compute:
    - abs(median - true)
    - wasserstein distance
    - coverage
    """
    samples = np.asarray(samples)
    median_est = np.median(samples)
    abs_err_median = abs(median_est - true_val)
    rel_err_median = 100 * abs_err_median / true_val
    wd = wasserstein_distance(samples, [true_val])
    cov = coverage(samples, true_val)
    return [abs_err_median, rel_err_median, wd, cov]