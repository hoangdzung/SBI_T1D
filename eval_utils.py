import numpy as np
from scipy.stats import wasserstein_distance

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