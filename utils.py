from typing import Any, Callable, Optional
import torch
import pyro.distributions as dist

from py_replay_bg.model.t1d_model_single_meal import T1DModelSingleMeal
from py_replay_bg.environment import Environment
from py_replay_bg.data import ReplayBGData

import pandas as pd 
import numpy as np 
from copy import deepcopy
from pathos.multiprocessing import ProcessPool as Pool
import random
from datetime import datetime, timedelta
import warnings

N_PARAMS=9
N_PARAMS_HAT=18
class CustomPrior:
    def __init__(self, VG: float = 1.45, device: Optional[Any] = None, 
                 fixed_beta:bool = False, bw: Optional[float] = None, 
                 u2ss: Optional[float] = None, old_prior = False):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed_beta = fixed_beta
        self.bw = bw
        self.u2ss = u2ss
        self.old_prior = old_prior
            
        self.VG = torch.tensor(VG, device=self.device)

        # Distributions
        self.gamma_SI = dist.Gamma(torch.tensor(3.3, device=self.device), torch.tensor(1 / 5e-4, device=self.device))
        self.normal_p2 = dist.Normal(torch.tensor(0.11, device=self.device), torch.tensor(0.004, device=self.device))
        if self.old_prior:
            self.normal_Gb = dist.Normal(torch.tensor(141.865, device=self.device), torch.tensor(8.70, device=self.device))
        else:
            self.normal_Gb = dist.Normal(torch.tensor(119.13, device=self.device), torch.tensor(7.11, device=self.device))


        self.Gb_low = torch.tensor(70.0, device=self.device)
        self.Gb_high = torch.tensor(180.0, device=self.device)

        self.ln_SG = dist.Normal(torch.tensor(-3.8, device=self.device), torch.tensor(0.5, device=self.device))
        if self.old_prior:
            self.ln_ka2 = dist.Normal(torch.tensor(-4.09676, device=self.device), torch.tensor(0.46674, device=self.device))
        else:
            self.ln_ka2 = dist.Normal(torch.tensor(-4.2875, device=self.device), torch.tensor(0.4274, device=self.device))
        if self.old_prior:
            self.ln_kd = dist.Normal(torch.tensor(-4.12588, device=self.device), torch.tensor(0.15242, device=self.device))            
        else:
            self.ln_kd = dist.Normal(torch.tensor(-3.5090, device=self.device), torch.tensor(0.6187, device=self.device))

        self.ln_kempt = dist.Normal(torch.tensor(-1.9646, device=self.device), torch.tensor(0.7069, device=self.device))
        if self.old_prior:
            self.ln_kabs = dist.Normal(torch.tensor(-1.99872, device=self.device), torch.tensor(1.50196, device=self.device))
        else:
            self.ln_kabs = dist.Normal(torch.tensor(-5.4591, device=self.device), torch.tensor(1.4396, device=self.device))

        
        # Additional params
        self.ln_u2ss = dist.Normal(torch.tensor(0.32477888, device=self.device), torch.tensor(0.36849, device=self.device))
        self.uni_BW = dist.Uniform(torch.tensor(20.0, device=self.device), torch.tensor(120.0, device=self.device))
        self.lower_bound, self.upper_bound = None, None
        # if self.fixed_beta:
        #     self.lower_bound = torch.tensor([70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0], device=self.device)
        #     self.upper_bound = torch.tensor([180.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 120.0], device=self.device)
        # else:
        #     self.lower_bound = torch.tensor([70.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0], device=self.device)
        #     self.upper_bound = torch.tensor([180.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 3.0, 120.0], device=self.device)
    
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
        SG = self.truncated_lognormal(-3.8, 0.5, low=1e-6, high=1.0, size=(n_samples,))
        if self.old_prior:
            Gb = self.truncated_normal(141.865, 8.70, low=70, high=180, size=(n_samples,))
            ka2 = self.truncated_lognormal(-4.09676, 0.46674, low=0.0, high=1.0, size=(n_samples,))
            kd = self.truncated_lognormal(-4.12588, 0.15242, low=0.0, high=1.0, size=(n_samples,))
            kempt = self.truncated_lognormal(-1.9646, 0.7069, low=0.0, high=1.0, size=(n_samples,))
            kabs = self.truncated_lognormal(-1.99872, 1.50196, low=0.0, high=1.0, size=(n_samples,))
        else:
            Gb = self.truncated_normal(119.13, 7.11, low=70, high=180, size=(n_samples,))
            ka2 = self.truncated_lognormal(-4.2875, 0.4274, low=1e-5, high=1.0, size=(n_samples,))
            kd = self.truncated_lognormal(-3.5090, 0.6187, low=ka2.min().item(), high=1.0, size=(n_samples,))
            mask_ka2 = ka2 < kd
            while not mask_ka2.all():
                ka2[~mask_ka2] = self.truncated_lognormal(-4.2875, 0.4274, 1e-5, 1.0, size=((~mask_ka2).sum().item(),))
                kd[~mask_ka2] = self.truncated_lognormal(-3.5090, 0.6187, ka2[~mask_ka2].min().item(), 1.0, size=((~mask_ka2).sum().item(),))
                mask_ka2 = ka2 < kd

            kempt = self.truncated_lognormal(-1.9646, 0.7069, low=1e-5, high=1.0, size=(n_samples,))
            kabs = self.truncated_lognormal(-5.4591, 1.4396, low=1e-6, high=kempt.min().item(), size=(n_samples,))
            # kabs = self.truncated_lognormal(-5.4591, 1.4396, low=1e-6, high=1.0, size=(n_samples,))
            # mask_kabs = kabs < kempt
            # while not mask_kabs.all():
            #     kempt[~mask_kabs] = self.truncated_lognormal(-1.9646, 0.7069, low=1e-5, high=1.0, size=((~mask_kabs).sum().item(),))
            #     kabs[~mask_kabs] = self.truncated_lognormal(-5.4591, 1.4396, kempt[~mask_kabs].min().item(), 1.0, size=((~mask_kabs).sum().item(),))
            #     mask_kabs = kabs < kempt

        if self.u2ss is None:
            u2ss = self.truncated_lognormal(0.32477888, 0.36849, low=0.0, high=3.0, size=(n_samples,))
        else:
            u2ss = torch.ones_like(ka2) * self.u2ss
        if self.bw is None:
            bw = self.uni_BW.sample((n_samples,))
        else:
            bw = torch.ones_like(ka2) * self.bw

        if self.fixed_beta:
            theta = torch.stack([Gb, SG, p2, ka2, kd, kempt, SI, kabs, u2ss, bw], dim=1)
        else:
            beta = torch.rand(n_samples, device=self.device) * 60
            theta = torch.stack([Gb, SG, p2, ka2, kd, kempt, SI, kabs, beta, u2ss, bw], dim=1)
        if squeeze:
            return theta.squeeze(0)
        return theta

    def log_prob(self, theta):
        if self.fixed_beta:
            Gb, SG, p2, ka2, kd, kempt, SI, kabs, u2ss, bw = theta.unbind(dim=1)
        else:
            Gb, SG, p2, ka2, kd, kempt, SI, kabs, beta, u2ss, bw = theta.unbind(dim=1)

        log_prob = torch.zeros(theta.shape[0], device=self.device)

        log_prob += self.gamma_SI.log_prob(SI * self.VG) + torch.log(self.VG)
        p2_sqrt = torch.sqrt(p2)
        log_prob += self.normal_p2.log_prob(p2_sqrt) - torch.log(2 * p2_sqrt)
        log_prob += self.normal_Gb.log_prob(Gb) - torch.log(self.normal_Gb.cdf(self.Gb_high) - self.normal_Gb.cdf(self.Gb_low))
        log_prob += self.ln_SG.log_prob(torch.log(SG)) - torch.log(SG)
        log_prob += self.ln_ka2.log_prob(torch.log(ka2)) - torch.log(ka2)
        log_prob += self.ln_kd.log_prob(torch.log(kd)) - torch.log(kd)
        log_prob += self.ln_kempt.log_prob(torch.log(kempt)) - torch.log(kempt)
        log_prob += self.ln_kabs.log_prob(torch.log(u2ss)) - torch.log(u2ss)
        log_prob += -torch.log(torch.tensor(100.0, device=self.device)) # bw
        log_prob += -torch.log(torch.tensor(60.0, device=self.device)) # beta

        if not self.old_prior:
            # Constraint: ka2 < kd, kabs < kempt
            mask_invalid = (ka2 >= kd) | (kabs >= kempt)
            
            log_prob[mask_invalid] = -float('inf')

        return log_prob
    
    # def mean(self):
    #     """
    #     Return mean tensors of the prior distribution,
    #     depending on self.fixed_beta and self.old_prior settings.
    #     """

    #     if self.old_prior:
    #         means = [
    #             141.865,   # Gb (Normal)
    #             0.0250,    # SG ~ LogNormal(-3.8, 0.5)
    #             0.01212,   # p2
    #             0.0168,    # ka2 ~ LogNormal(-4.09676, 0.46674)
    #             0.0162,    # kd ~ LogNormal(-4.12588, 0.15242)
    #             0.180,     # kempt
    #             0.00114,   # SI ~ Gamma(3.3, 1/5e-4)/VG
    #             0.135,     # kabs ~ LogNormal(-1.99872, 1.50196)
    #         ]
    #     else:
    #         means = [
    #             119.13,    # Gb (Normal)
    #             0.0250,    # SG
    #             0.01212,   # p2
    #             0.0151,    # ka2
    #             0.0360,    # kd
    #             0.180,     # kempt
    #             0.00114,   # SI
    #             0.0120,    # kabs
    #         ]

    #     # Handle beta
    #     if not self.fixed_beta:
    #         means.append(30.0)   # Uniform(0, 60)

    #     # Add u2ss and bw
    #     means.extend([1.48, 70.0])   # u2ss, bw

    #     return torch.tensor(means, device=self.device)

    # def variance(self):
    #     """
    #     Return variance tensors of the prior distribution,
    #     depending on self.fixed_beta and self.old_prior settings.
    #     """

    #     if self.old_prior:
    #         stds = [
    #             8.70,      # Gb
    #             0.0137,    # SG
    #             0.00088,   # p2
    #             0.0078,    # ka2
    #             0.0025,    # kd
    #             0.144,     # kempt
    #             0.000198,  # SI
    #             0.255,     # kabs
    #         ]
    #     else:
    #         stds = [
    #             7.11,      # Gb
    #             0.0137,    # SG
    #             0.00088,   # p2
    #             0.0067,    # ka2
    #             0.0240,    # kd
    #             0.144,     # kempt
    #             0.000198,  # SI
    #             0.0320,    # kabs
    #         ]

    #     # Handle beta
    #     if not self.fixed_beta:
    #         stds.append(17.32)

    #     # Add u2ss and bw
    #     stds.extend([0.57, 28.87])   # u2ss, bw

    #     # Convert std -> variance
    #     vars_ = [s**2 for s in stds]

    #     return torch.tensor(vars_, device=self.device)
   
def get_prior(VG: float = 1.45, 
            device = torch.device('cpu'), 
            fixed_beta=False, bw=None, u2ss=None, old_prior=False) -> CustomPrior:
    custom_prior = CustomPrior(VG=VG, device=device, 
                            fixed_beta=fixed_beta, 
                            bw=bw, u2ss=u2ss, old_prior=old_prior)
    return custom_prior

def simulate_one(args):
    model, *params = args
    return model.sbi_simulate(*params)

def sample_and_simulate_one(args):
    bw, u2ss, fixed_beta, meal_gen_funct, *params = args
    new_df = meal_gen_funct(minute_interval=1)
    model, rbg_data = get_model_and_rbg_data(new_df, bw, u2ss, fixed_beta=fixed_beta,
                                            rbg_data_kwargs={"bolus_source": "dss", "basal_source": "dss"})
    return model.sbi_simulate(rbg_data, *params)



def get_model_and_rbg_data(data, bw, u2ss, glucose_sequence=None, cho=None, fixed_beta=False, rbg_data_kwargs={}):
    if type(data) is str:
        data = pd.read_csv(data)
        data.t = pd.to_datetime(data['t'])
    elif isinstance(data, pd.DataFrame):
        data = deepcopy(data)
    else:
        raise TypeError("Data must be a file path or a pandas DataFrame.")
    if glucose_sequence is not None:
        common_len = min(len(data['glucose']), len(glucose_sequence))
        data['glucose'].values[:common_len] = glucose_sequence[:common_len]
    if cho is not None:
        data.cho = cho
 
    # Environment and model setup
    env = Environment(save_name='ori', save_folder='./')
    model = T1DModelSingleMeal(data=data, bw=bw, u2ss=u2ss, environment=env, fixed_values={'beta':0.0} if fixed_beta else {})
    rbg_data = ReplayBGData(data=data, model=model, environment=env, **rbg_data_kwargs)
    return model, rbg_data

def create_patient_df(t, glucose, cho, bolus, basal):
    # Create empty label columns

    n = len(bolus)

    if len(glucose) != n:
        # Assume glucose is measured every k minutes
        k = n // len(glucose)
        if n % len(glucose) != 0:
            raise ValueError("Glucose length must divide evenly into bolus length.")

        # Expand glucose: each reading followed by k-1 NaNs
        glucose_extended = np.full(n, np.nan, dtype=float)
        glucose_extended[::k] = glucose
        glucose = glucose_extended

    bolus_label = np.full_like(bolus, np.nan, dtype=float)
    cho_label = np.full_like(cho, np.nan, dtype=float)

    # Build DataFrame
    df = pd.DataFrame({
        "t": pd.to_datetime(t),
        "glucose": glucose,
        "cho": cho,
        "bolus": bolus,
        "basal": basal,
        "bolus_label": bolus_label,
        "cho_label": cho_label
    })

    return df


def simglucose_basal_handler(
    glucose: np.ndarray,
    meal_announcement: np.ndarray,
    meal_type: np.ndarray,
    hypotreatments: np.ndarray,
    bolus: np.ndarray,
    basal: np.ndarray,
    time: np.ndarray,
    time_index: int,
    dss: object
    ) -> tuple[float, object]:

    # If G < 70...
    if glucose[time_index] < 70:
        # ...set basal rate to 0
        b = 0
    else:
        basal_handler_params = getattr(dss, 'basal_handler_params', {})
        u2ss = basal_handler_params.get('u2ss', 1.43)
        b = u2ss / 6
    return b, dss


def simglucose_bolus_calculator_handler(
        glucose: np.ndarray,
        meal_announcement: np.ndarray,
        meal_type: np.ndarray,
        hypotreatments: np.ndarray,
        bolus: np.ndarray,
        basal: np.ndarray,
        time: np.ndarray,
        time_index: int,
        dss: object
        ) -> tuple[float, object]:

    b = 0

    if meal_announcement[time_index] > 0:
        bolus_params = getattr(dss, 'bolus_calculator_handler_params', {})
        cr = bolus_params.get('cr', 10)
        cf = bolus_params.get('cf', 40)
        gt = bolus_params.get('gt', 120)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                b = np.max([
                    0,
                    meal_announcement[time_index] / cr + (glucose[time_index] - gt) / cf
                ])
            except RuntimeWarning as w:
                raise RuntimeError(
                    f"Bolus calculation warning at time_index={time_index}: {w}\n"
                    f"(meal={meal_announcement[time_index]}, glucose={glucose[time_index]}, "
                    f"cr={cr}, cf={cf}, gt={gt})"
                ) from w
        # print(f"Simglucose bolus: Meal at {time[time_index]}: carbs={meal_announcement[time_index]}, glucose={glucose[time_index]}, bolus={b}")

    return b, dss

# https://scispace.com/pdf/design-and-validation-of-an-open-source-closed-loop-testbed-3vn3wol7.pdf
def virginia_bolus_calculator_handler(
        glucose: np.ndarray,
        meal_announcement: np.ndarray,
        meal_type: np.ndarray,
        hypotreatments: np.ndarray,
        bolus: np.ndarray,
        basal: np.ndarray,
        time: np.ndarray,
        time_index: int,
        dss: object
        ) -> tuple[float, object]:
    """
    Implements the default bolus calculator formula: B = CHO/CR + (GC-GT)/CF - IOB

    Parameters
    ----------
    glucose: np.ndarray
        An array vector as long the simulation length containing all the simulated glucose concentrations (mg/dl)
        up to time_index. The values after time_index should be ignored.
    meal_announcement: np.ndarray
        An array vector as long the simulation length containing all the meal announcements (g) up to time_index.
        The values after time_index should be ignored.
    meal_type: np.ndarray
        An array of strings as long the simulation length containing the type of each meal.
        If blueprint is `single-meal`, labels can be:
            - `M`: main meal
            - `O`: other meal
        If blueprint is `multi-meal`, labels can be:
            - `B`: breakfast
            - `L`: lunch
            - `D`: dinner
            - `S`: snack
            - `H`: hypotreatment
        The values after time_index should be ignored.
    hypotreatments: np.ndarray
        An array vector as long the simulation length containing all the hypotreatment intakes (g/min) up to time_index.
        If the blueprint is single meal, hypotreatments will contain only the hypotreatments generated by this function
        during the simulation. If the blueprint is multi-meal, hypotreatments will ALSO contain the hypotreatments
        already present in the given data that labeled as such. The values after time_index should be ignored.
    bolus: np.ndarray
        An array vector as long the simulation length containing all the insulin boluses (U/min) up to time_index.
        The values after time_index should be ignored.
    basal: np.ndarray
        An array vector as long the simulation length containing all the insulin basal (U/min) up to time_index.
        The values after time_index should be ignored.
    time: np.ndarray
        An array vector as long the simulation length containing the time corresponding to the current step (hours) up
        to time_index. The values after time_index should be ignored.
    time_index: int
        The index corresponding to the previous simulation step of the replay simulation.
    dss: DSS
        An object that represents the hyperparameters of the integrated decision support system.

    Returns
    -------
    b: float
        The bolus insulin rate to administer at time[time_index+1].
    dss: DSS
        An object that represents the hyperparameters of the integrated decision support system.
        dss is also an output since it contains bolus_calculator_handler_params that beside being a
        dict that contains the parameters to pass to  this function, it also serves as memory area.
        It is possible to store values inside it and the standard_bolus_calculator_handler function will be able
        to access to them in the next call of the function.

    Raises
    ------
    None

    See Also
    --------
    None

    Examples
    --------
    None
    """

    b = 0

    # If a meal is announced...
    if meal_announcement[time_index] > 0:

        # compute iob
        ts = 5

        k1 = 0.0173
        k2 = 0.0116
        k3 = 6.73

        iob_6h_curve = np.zeros(shape=(360,))

        for t in range(0, 360):
            iob_6h_curve[t] = 1 - 0.75 * ((- k3 / (k2 * (k1 - k2)) * (np.exp(-k2 * t / 0.75) - 1) + k3 / (
                        k1 * (k1 - k2)) * (np.exp(-k1 * t / 0.75) - 1)) / 2.4947e4)
        iob_6h_curve = iob_6h_curve[ts::ts]

        iob = np.convolve(bolus, iob_6h_curve)
        iob = iob[bolus.shape[0] - 1]

        # get params
        bolus_params = getattr(dss, 'bolus_calculator_handler_params', {})
        if 'bw' in bolus_params:
            tdd = 0.55 * bolus_params['bw']  # total daily dose
            cr = 450 / tdd
            cf = 1700 / tdd
        else:
            cr = bolus_params['cr'] if 'cr' in bolus_params else 10
            cf = bolus_params['cf'] if 'cf' in bolus_params else 40
        gt = bolus_params['gt'] if 'gt' in bolus_params else 120

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                # ...give a bolus
                b = np.max([0, meal_announcement[time_index] / cr + (glucose[time_index] - gt) / cf - iob])
            except RuntimeWarning as w:
                raise RuntimeError(
                    f"Bolus calculation warning at time_index={time_index}: {w}\n"
                    f"(meal={meal_announcement[time_index]}, glucose={glucose[time_index]}, "
                    f"cr={cr}, cf={cf}, gt={gt})"
                ) from w
        # print(f"Standard bolus: Meal at {time[time_index]}: carbs={meal_announcement[time_index]}, glucose={glucose[time_index]}, bolus={b}")

    return b, dss

def sample_meals_for_3_days(minute_interval=5):
    def random_time(hour_range, minute_interval=minute_interval):
        hour = random.randint(hour_range[0], hour_range[1])
        minute = random.randint(0, (60 // minute_interval) - 1) * minute_interval
        return f"{hour:02d}:{minute:02d}:00"  # seconds fixed to 00
    
    # Store only meals first
    meal_entries = {}
    
    for day_offset in range(3):
        date = datetime.today().date() + timedelta(days=day_offset)
        
        meals = [
            {"time": random_time((6, 9)),  "carbs": random.randint(5, 10), "label": "B"},
            {"time": random_time((11, 14)), "carbs": random.randint(10, 20), "label": "L"},
            {"time": random_time((17, 21)), "carbs": random.randint(10, 20), "label": "D"}
        ]
        
        # Random snacks
        for _ in range(random.randint(0, 0)):
            meals.append({
                "time": random_time((8, 22)),
                "carbs": random.randint(5, 10),
                "label": "S"
            })
        
        # Store in dictionary keyed by datetime
        for meal in meals:
            dt = datetime.strptime(f"{date} {meal['time']}", "%Y-%m-%d %H:%M:%S")
            meal_entries[dt] = {
                "cho": meal["carbs"],
                "cho_label": meal["label"]
            }
    
    # Build full 5-minute frequency index for 3 days
    start_time = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=3) - timedelta(minutes=minute_interval)
    full_index = pd.date_range(start=start_time, end=end_time, freq=f"{minute_interval:d}min")
    
    # Create dataframe with default values
    df = pd.DataFrame({
        "t": full_index,
        "cho": 0,
        "cho_label": pd.Series([np.nan] * len(full_index), dtype="object")     # force object dtype

    })
    
    # Fill in meal entries
    for t, vals in meal_entries.items():
        mask = df["t"] == t
        for k, v in vals.items():
            df.loc[mask, k] = v
    
    return df

# Helper to index ragged arrays safely
def split_array(arr, idx):
    return arr[idx]

def first_half(arr):
    """Return the first half of elements (works for 1D or 2D arrays)."""
    if arr.ndim == 1:  # 1D array
        return arr[: arr.shape[0] // 2]
    elif arr.ndim == 2:  # 2D array
        return arr[:, : arr.shape[1] // 2]
    else:
        raise ValueError("Input must be 1D or 2D array")
    
def second_half(arr):
    """Return the second half of elements (works for 1D or 2D arrays)."""
    if arr.ndim == 1:  # 1D array
        return arr[-arr.shape[0] // 2:]
    elif arr.ndim == 2:  # 2D array
        return arr[:, -arr.shape[1] // 2:]
    else:
        raise ValueError("Input must be 1D or 2D array")
    
def parse_kv(s):
    key, value = s.split('=')
    return key, float(value)  # or int(value) if appropriate