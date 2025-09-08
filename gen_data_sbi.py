import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd
from pathos.multiprocessing import ProcessPool as Pool

from utils import sample_and_simulate_one, simulate_one, get_prior, split_array, first_half, virginia_bolus_calculator_handler, simglucose_basal_handler, sample_meals_for_3_days
from sbi.utils import RestrictionEstimator
from sbi.utils.user_input_checks import process_prior
from py_replay_bg.dss import DSS

def main(args):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.patient_info_path is not None and os.path.isfile(args.patient_info_path):
        patient_info = pd.read_csv(args.patient_info_path)
        p = np.where(patient_info['patient'] == 1)[0][0]
        bw = float(patient_info.bw.values[p])
        u2ss = float(patient_info.u2ss.values[p])
    else:
        print("No patient info file provided or file does not exist, sampling bw and u2ss from prior")
        # otherwise we sample
        bw, u2ss = None, None

    custom_prior = get_prior(device=device, 
                            fixed_beta=args.fixed_beta,
                            bw=bw, u2ss=u2ss, old_prior=args.old_prior)
    prior, _, _ = process_prior(custom_prior)

    restriction_estimator = RestrictionEstimator(prior=prior, decision_criterion=lambda x: x.squeeze())
    proposals = [prior]
     
    all_ts, all_x0s, all_thetas, all_cgms, all_boluses, all_basals, all_meals\
        = [], [], [], [], [], [], []
    # add tqdm for while loop
    pbar = tqdm(total=args.num_train + args.num_test, desc="Generating samples", leave=True)
    while len(all_x0s) < args.num_train + args.num_test:
        theta = proposals[-1].sample((args.batch_size,)).to(device)

        # Run simulations in parallel
        if args.meal_sampling:
            input_data = []
            for i in range(args.batch_size):
                bw = theta[i][-1].item()
                u2ss = theta[i][-2].item()
                dss = DSS(bw=bw, enable_hypotreatments=True, 
                        enable_correction_boluses=False, 
                        basal_handler=simglucose_basal_handler, 
                        bolus_calculator_handler=virginia_bolus_calculator_handler,
                        basal_handler_params = {'BW': bw, 'u2ss': u2ss},
                        bolus_calculator_handler_params = {'BW': bw})
                
                x0 = None
                input_data.append((bw, u2ss, args.fixed_beta, sample_meals_for_3_days, theta[i][:-2].cpu().numpy(), dss, x0, args.num_hour, True))
                
            simulate_funct = sample_and_simulate_one
        else:
            raise NotImplementedError("Not implemented for no meal sampling yet")
            dss, x0 = None, None
            input_data = [(model, rbg_data, theta[i][:-2].cpu().numpy(), dss, x0, args.num_hour, True) for i in range(args.batch_size)]
            simulate_funct = simulate_one
        
        if args.sequential:
            results = []
            for i in tqdm(input_data):
                result = simulate_funct(i)
                t, x, cgm, bolus, basal, meal = result
                results.append(result)
        else:
            with Pool() as pool:
                results = pool.map(simulate_funct, input_data)

        # Filter valid results
        batch_fake_x = []
        for i, (t, x, cgm, bolus, basal, meal) in enumerate(results):
            if cgm is not None and cgm.max() <= 400 and cgm.min() >= 40 \
                and (x[0] >=0).all() and (x[2:] >=0).all() and (bolus >=0).all() and (basal >=0).all():
                all_ts.append(t)
                all_x0s.append(x[:, 0])
                all_thetas.append(theta[i].cpu().numpy())
                all_cgms.append(cgm)
                all_boluses.append(bolus)
                all_basals.append(basal)
                all_meals.append(meal)
                batch_fake_x.append([1.0])
                pbar.update(1)
            else:
                batch_fake_x.append([0.0]) 
        batch_fake_x = torch.tensor(batch_fake_x, device=device)
        print(f"Generated {len(batch_fake_x)} samples in this batch, valid: {batch_fake_x.sum()}")
        
        if batch_fake_x.sum() < 2:
            print("Not enough valid samples generated, skipping restriction estimation.")
            continue
        
        # Sample theta and batch_fake_x such that the number of invalid samples is twice the number of valid samples
        valid_indices = (batch_fake_x.squeeze() == 1.0).nonzero(as_tuple=True)[0]
        invalid_indices = (batch_fake_x.squeeze() == 0.0).nonzero(as_tuple=True)[0]

        num_valid = len(valid_indices)
        num_invalid_needed = num_valid 

        if len(invalid_indices) > num_invalid_needed:
            # Randomly sample the required number of invalid samples
            invalid_indices = invalid_indices[torch.randperm(len(invalid_indices))[:num_invalid_needed]]

        # Combine selected valid and invalid indices
        selected_indices = torch.cat([valid_indices, invalid_indices], dim=0)

        # Shuffle the selected indices
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        # Filter theta and batch_fake_x according to the selection
        sampled_theta = theta[selected_indices]
        sampled_batch_fake_x = batch_fake_x[selected_indices]   

        restriction_estimator.append_simulations(sampled_theta, sampled_batch_fake_x)
        _ = restriction_estimator.train()
        # try:
        proposal = restriction_estimator.restrict_prior()
        proposals.append(proposal)
        # except Exception as e:
        #     # TODO: fix this
        #     print(f"Error during restriction estimation: {e}")
        #     continue

    all_ts = np.stack(all_ts, axis=0)       # shape: (N, T)
    all_x0s = np.stack(all_x0s, axis=0)     # shape: (N, features)
    all_thetas = np.stack(all_thetas, axis=0)
    all_cgms = np.stack(all_cgms, axis=0)
    all_boluses = np.stack(all_boluses, axis=0)
    all_basals = np.stack(all_basals, axis=0)
    all_meals = np.stack(all_meals, axis=0)

    # Ensure same length
    assert len(all_ts) == len(all_x0s) == len(all_thetas) == len(all_cgms) == len(all_boluses) == len(all_basals) == len(all_meals), \
        "Data length mismatch — lists must be the same length."

    # Create shuffled indices
    indices = np.random.permutation(len(all_x0s))
    train_idx = indices[:args.num_train]
    test_idx = indices[args.num_train:]

    # Split datasets
    train_data = {
        "t": split_array(all_ts, train_idx),
        "x0s": split_array(all_x0s, train_idx),
        "thetas": split_array(all_thetas, train_idx),
        "cgms": first_half(split_array(all_cgms, train_idx)),
        "boluses": first_half(split_array(all_boluses, train_idx)),
        "basals": first_half(split_array(all_basals, train_idx)),
        "meals": first_half(split_array(all_meals, train_idx)),
    }

    test_data = {
        "t": split_array(all_ts, test_idx),
        "x0s": split_array(all_x0s, test_idx),
        "thetas": split_array(all_thetas, test_idx),
        "cgms": split_array(all_cgms, test_idx),
        "boluses": split_array(all_boluses, test_idx),
        "basals": split_array(all_basals, test_idx),
        "meals": split_array(all_meals, test_idx)
    }

    # Save
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(train_data, os.path.join(args.save_path, "train_data.pt"))
    torch.save(test_data, os.path.join(args.save_path, "test_data.pt"))

    print("Saved train/test splits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SBI with Restriction Estimator and Save Fixed Dataset Split")
    parser.add_argument("--data_path", type=str, default='./data/data_day_1.csv', help="Path to the csv data file")
    parser.add_argument("--patient_info_path", type=str, help="Path to the patient_info csv data file")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size per inference round")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training samples to save")
    parser.add_argument("--num_test", type=int, default=50, help="Number of test samples to save")
    parser.add_argument("--save_path", type=str, default="./data/simulated", help="Disable CUDA and use CPU even if available")
    parser.add_argument("--fixed_beta", action="store_true", help="Whether to fix beta as 0")
    parser.add_argument("--old_prior", action="store_true", help="Whether to use simglucose old prior")
    parser.add_argument("--meal_sampling", action="store_true", help="Whether to sample meal")
    parser.add_argument("--num_hour", type=int, default=24, help="Number of window size in hours for inference")
    parser.add_argument("--sequential", action="store_true", help="Whether to run simulation sequentially instead of parallel")
    args = parser.parse_args()
    main(args)
