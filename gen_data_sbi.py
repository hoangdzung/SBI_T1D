import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool as Pool

from utils import simulate_one, get_prior, get_model_and_rbg_data
from sbi.utils import RestrictionEstimator
from sbi.utils.user_input_checks import process_prior


def main(args):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, rbg_data = get_model_and_rbg_data(args.data_path, args.patient_info_path, fixed_beta=args.fixed_beta)
    if args.meal_sampling:
        sample_meal_func = lambda : None # TODO: Implement meal sampling function
    else:
        sample_meal_func = None
    custom_prior = get_prior(model, rbg_data, device, sample_meal_func, args.fixed_beta)
    prior, _, _ = process_prior(custom_prior)

    restriction_estimator = RestrictionEstimator(prior=prior)
    proposals = [prior]

    for r in tqdm(range(args.num_rounds), desc="SBI Rounds"):
        # TODO: Somehow also need to return sampled meal here
        sampled_meal = None
        theta = proposals[-1].sample((args.batch_size,)).to(device)
        print(theta.shape)

        # Prepare inputs for multiprocessing
        # TODO: update rbg_data with sampled meal
        input_data = [(theta[i].cpu().numpy(), model, rbg_data) for i in range(args.batch_size)]

        # Run simulations in parallel
        # TODO: Simulate_one also need to return insulin if meal sampling is enabled
        with Pool() as pool:
            results = pool.map(simulate_one, input_data)

        # Filter valid results
        valid_idxs = []
        xs = []
        for i, result in enumerate(results):
            if result[1] is not None:
                valid_idxs.append(i)
                xs.append(result[1])

        if len(xs) == 0:
            print(f"Warning: No valid simulations in round {r}. Skipping...")
            continue

        x = torch.from_numpy(np.stack(xs)).float().to(device)
        theta = theta[valid_idxs].float()

        # Apply value filtering
        x[(x < 40) | (x > 400)] = float("nan")

        restriction_estimator.append_simulations(theta, x)

        # Train classifier except on last round
        if r < args.num_rounds - 1:
            _ = restriction_estimator.train()

        proposal = restriction_estimator.restrict_prior()
        proposals.append(proposal)

    # Retrieve all simulations
    all_theta, all_x, _ = restriction_estimator.get_simulations()
    print("Before filtering:", all_theta.shape, all_x.shape)

    # Filter out NaNs
    valid_rows = (
        (~torch.isnan(all_theta).any(dim=1)) &
        (~torch.isnan(all_x).any(dim=1)) &
        (all_x.ge(40).all(dim=1)) &
        (all_x.le(400).all(dim=1))
    )
    all_theta = all_theta[valid_rows]
    all_x = all_x[valid_rows]
    print("After filtering:", all_theta.shape, all_x.shape)

    # Ensure enough valid samples for requested split
    min_test_samples = args.num_test
    if all_theta.shape[0] < min_test_samples:
        raise ValueError(
            f"Not enough valid samples ({all_theta.shape[0]}) to allocate even {min_test_samples} test samples."
        )

    # If not enough for full train+test split, warn and use what we have
    total_available = all_theta.shape[0]
    total_needed = args.num_train + args.num_test

    if total_available < total_needed:
        print(f"⚠️ Warning: Not enough data for full split. Requested {args.num_train} train + {args.num_test} test, "
            f"but only {total_available} valid samples available.")
        print(f"Using {total_available - args.num_test} for training and {args.num_test} for testing.")

    # Shuffle and split
    indices = torch.randperm(total_available)
    test_idx = indices[:args.num_test]
    train_idx = indices[args.num_test:][:args.num_train]

    train_theta, train_x = all_theta[train_idx], all_x[train_idx]
    test_theta, test_x = all_theta[test_idx], all_x[test_idx]

    # Save splits
    os.makedirs(args.save_path, exist_ok=True)
    torch.save([train_theta, train_x], os.path.join(args.save_path, "train_data.pt"))
    torch.save([test_theta, test_x], os.path.join(args.save_path, "test_data.pt"))

    print("Saved train/test splits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SBI with Restriction Estimator and Save Fixed Dataset Split")
    parser.add_argument("--data_path", type=str, default='./data/data_day_1.csv', help="Path to the csv data file")
    parser.add_argument("--patient_info_path", type=str, default='./data/patient_info.csv', help="Path to the patient_info csv data file")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size per inference round")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of SBI rounds")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training samples to save")
    parser.add_argument("--num_test", type=int, default=50, help="Number of test samples to save")
    parser.add_argument("--save_path", type=str, default="./data/simulated", help="Disable CUDA and use CPU even if available")
    parser.add_argument("--fixed_beta",action="store_true", help="Whether to fix beta as 0")
    parser.add_argument("--meal_sampling",action="store_true", help="Whether to sample meal")
    args = parser.parse_args()
    main(args)
