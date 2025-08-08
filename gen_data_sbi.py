import torch

import os
import argparse
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool as Pool

from utils import simulate_one, get_prior, get_model_and_rbg_data
from sbi.utils import RestrictionEstimator
from sbi.utils.user_input_checks import process_prior
from py_replay_bg.dss import DSS

def main(args):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, rbg_data = get_model_and_rbg_data(args.data_path, args.patient_info_path, fixed_beta=args.fixed_beta)
    custom_prior = get_prior(VG=model.model_parameters.VG, 
                             device=device, 
                             fixed_beta=args.fixed_beta)
    prior, _, _ = process_prior(custom_prior)

    restriction_estimator = RestrictionEstimator(prior=prior, decision_criterion=lambda x: x.squeeze())
    proposals = [prior]
    if args.meal_sampling:
        dss = DSS(bw=model.model_parameters.bw, enable_hypotreatments=True, enable_correction_boluses=True)
    else:
        dss = None
    
    all_x0s, all_thetas, all_cgms, all_boluses, all_basals, all_meals = [], [], [], [], [], []
    # add tqdm for while loop
    pbar = tqdm(total=args.num_train + args.num_test, desc="Generating samples", leave=True)
    while len(all_x0s) < args.num_train + args.num_test:
        theta = proposals[-1].sample((args.batch_size,)).to(device)
        print(theta.shape)

        # Prepare inputs for multiprocessing
        input_data = [(theta[i].cpu().numpy(), model, rbg_data, dss, None) for i in range(args.batch_size)]

        # Run simulations in parallel
        with Pool() as pool:
            results = pool.map(simulate_one, input_data)

        # Filter valid results
        batch_fake_x = []
        for i, (x, cgm, bolus, basal, meal) in enumerate(results):
            if cgm is not None and cgm.max() <= 400 and cgm.min() >= 40 \
                and (bolus >=0).all() and (basal >=0).all():
                all_x0s.append(x[0])
                all_thetas.append(theta[i].cpu().numpy())
                all_cgms.append(cgm)
                all_boluses.append(bolus)
                all_basals.append(basal)
                all_meals.append(meal)
                all_thetas.append(theta[i].cpu().numpy())
                batch_fake_x.append([1.0])
                pbar.update(1)
            else:
                batch_fake_x.append([0.0])  
        batch_fake_x = torch.tensor(batch_fake_x, device=device)
        print(f"Generated {len(batch_fake_x)} samples in this batch, valid: {batch_fake_x.sum()}")
        
        if batch_fake_x.sum() == 0:
            print("No valid samples generated, skipping restriction estimation.")
            continue
        
        # Sample theta and batch_fake_x such that the number of invalid samples is twice the number of valid samples
        valid_indices = (batch_fake_x.squeeze() == 1.0).nonzero(as_tuple=True)[0]
        invalid_indices = (batch_fake_x.squeeze() == 0.0).nonzero(as_tuple=True)[0]

        num_valid = len(valid_indices)
        num_invalid_needed = num_valid * 2

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
        try:
            proposal = restriction_estimator.restrict_prior()
            proposals.append(proposal)
        except Exception as e:
            print(f"Error during restriction estimation: {e}")
            continue

    # Convert to torch tensors (list of tensors if shapes vary)
    def to_tensor_list(lst):
        return [torch.tensor(x, dtype=torch.float32) for x in lst]

    all_x0s = to_tensor_list(all_x0s)
    all_thetas = to_tensor_list(all_thetas)
    all_cgms = to_tensor_list(all_cgms)
    all_boluses = to_tensor_list(all_boluses)
    all_basals = to_tensor_list(all_basals)
    all_meals = to_tensor_list(all_meals)

    # Ensure same length
    assert len(all_x0s) == len(all_thetas) == len(all_cgms) == len(all_boluses) == len(all_basals) == len(all_meals), \
        "Data length mismatch — lists must be the same length."

    # Create index list and split
    indices = torch.randperm(len(all_x0s))
    split_idx = int(len(indices) * 0.8)  # 80% train, 20% test
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # Helper function to index lists
    def split_list(lst, idx):
        return [lst[i] for i in idx]

    # Split datasets
    train_data = {
        "x0s": split_list(all_x0s, train_idx),
        "thetas": split_list(all_thetas, train_idx),
        "cgms": split_list(all_cgms, train_idx),
        "boluses": split_list(all_boluses, train_idx),
        "basals": split_list(all_basals, train_idx),
        "meals": split_list(all_meals, train_idx)
    }

    test_data = {
        "x0s": split_list(all_x0s, test_idx),
        "thetas": split_list(all_thetas, test_idx),
        "cgms": split_list(all_cgms, test_idx),
        "boluses": split_list(all_boluses, test_idx),
        "basals": split_list(all_basals, test_idx),
        "meals": split_list(all_meals, test_idx)
    }

    os.makedirs(args.save_path, exist_ok=True)
    torch.save(train_data, os.path.join(args.save_path, "train_data.pt"))
    torch.save(test_data, os.path.join(args.save_path, "test_data.pt"))

    print("Saved train/test splits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SBI with Restriction Estimator and Save Fixed Dataset Split")
    parser.add_argument("--data_path", type=str, default='./data/data_day_1.csv', help="Path to the csv data file")
    parser.add_argument("--patient_info_path", type=str, default='./data/patient_info.csv', help="Path to the patient_info csv data file")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size per inference round")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training samples to save")
    parser.add_argument("--num_test", type=int, default=50, help="Number of test samples to save")
    parser.add_argument("--save_path", type=str, default="./data/simulated", help="Disable CUDA and use CPU even if available")
    parser.add_argument("--fixed_beta",action="store_true", help="Whether to fix beta as 0")
    parser.add_argument("--meal_sampling",action="store_true", help="Whether to sample meal")
    args = parser.parse_args()
    main(args)
