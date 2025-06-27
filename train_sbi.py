import os
import argparse
import torch
from sbi.inference import NPE
from sbi.utils.user_input_checks import process_prior
from utils import get_prior, get_model_and_rbg_data


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and prior
    model, rbg_data = get_model_and_rbg_data(args.data_path, args.patient_info_path, fixed_beta=args.fixed_beta)
    custom_prior = get_prior(model, rbg_data, device, fixed_beta=args.fixed_beta)
    prior, _, _ = process_prior(custom_prior)

    # Load training data
    train_theta, train_x = torch.load(args.train_data)
    train_theta, train_x = train_theta.to(device), train_x.to(device)
    # Inference
    inference = NPE(prior=prior, device=device)
    inference = inference.append_simulations(train_theta, train_x)
    density_estimator = inference.train()

    # Save trained model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(density_estimator, args.output_path)
    print(f"✅ Saved density estimator to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NPE using saved simulations.")
    parser.add_argument("--data_path", type=str, default='./data/data_day_1.csv', help="Path to the csv data file")
    parser.add_argument("--patient_info_path", type=str, default='./data/patient_info.csv', help="Path to the patient_info csv data file")
    parser.add_argument("--train_data", type=str, default='./data/simulated/train_data.pt', help="Path to training theta .pt file.")
    parser.add_argument("--output_path", type=str, default="./trained_models/density_estimator.pt")
    parser.add_argument("--fixed_beta",action="store_true", help="Whether to fix beta as 0")

    args = parser.parse_args()
    main(args)
