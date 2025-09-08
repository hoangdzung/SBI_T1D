import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
from multiprocessing import freeze_support
from py_replay_bg.py_replay_bg import ReplayBG
from utils import create_patient_df, first_half, parse_kv

def main(args):
    freeze_support()

    # Validate method
    if args.method not in ["map", "mcmc"]:
        raise ValueError(f"Invalid method: {args.method}. Must be 'map' or 'mcmc'.")

    # Load test data
    test_data = torch.load(args.test_data, weights_only=False)
    
    if args.index >= len(test_data["t"]):
        raise IndexError(f"Index {args.index} out of range for test data of size {len(test_data['t'])}")
    
    data = create_patient_df(t=first_half(test_data['t'][args.index]),
                             glucose=first_half(test_data['cgms'][args.index]),
                             cho=first_half(test_data['meals'][args.index]),
                             bolus=first_half(test_data['boluses'][args.index]),
                             basal=first_half(test_data['basals'][args.index]))
    # Load patient info
    patient_info = pd.read_csv(args.patient_info_path)
    p_idx = np.where(patient_info['patient'] == 1)[0][0]
    bw = float(patient_info.bw.values[p_idx])
    u2ss = float(patient_info.u2ss.values[p_idx])

    save_name = f"{args.method}_data_day_1_{args.index}"
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    # Initialize ReplayBG
    rbg = ReplayBG(
        blueprint='single-meal',
        save_folder=save_folder,
        yts=5,
        exercise=False,
        seed=1,
        verbose=True,
        plot_mode=False,
        fixed_values=dict(args.fixed_values),
    )

    # Step 1: Twinning
    print(f"⏳ Starting twinning with method: {args.method}")
    start_time = time.time()
    rbg.twin(
        data=data,
        bw=bw,
        save_name=save_name,
        twinning_method=args.method,
        parallelize=True,
        n_steps=5000,
        u2ss=u2ss
    )
    print(f"✅ Twinning completed in {time.time() - start_time:.2f} seconds.")

    # Step 2: Replay
    print("⏳ Starting replay...")
    start_time = time.time()
    rbg.replay(
        data=data,
        bw=bw,
        save_name=save_name,
        twinning_method=args.method,
        save_workspace=True,
        save_suffix='_step_2a'
    )
    print(f"✅ Replay completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReplayBG twinning and replay.")
    parser.add_argument("--patient_info_path", type=str, default='./data/patient_info.csv', help="Path to the patient_info csv data file")
    parser.add_argument("--index", type=int, required=True, help="Index of test sample to use.")
    parser.add_argument("--method", type=str, required=True, choices=["map", "mcmc"],
                        help="Twinning method to use: 'map' or 'mcmc'.")
    parser.add_argument("--test_data", type=str, default='./data/simulated/test_data.pt', help="Path to testing theta .pt file.")
    parser.add_argument("--noise_meal", action="store_true", help="Whether to add noise to meal announcements.")
    parser.add_argument("--save_folder", type=str, default='./', help="Path to save results.")
    parser.add_argument('--fixed_values', type=parse_kv, nargs='+', help='Parameters to be fixed')

    args = parser.parse_args()
    main(args)
