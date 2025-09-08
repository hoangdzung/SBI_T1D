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
    
    data = pd.read_csv(args.test_data).iloc[:-1]
    data = data.rename(columns={"Time": "t", "BG": "glucose", "CHO":"cho"})
    data['t'] = pd.to_datetime(data['t'])
    data[['bolus_label', 'basal_label']] = None

    # Load patient info
    patient_id = os.path.splitext(os.path.basename(args.test_data))[0]
    patient_info = pd.read_csv('/storage/homefs/th24p772/.conda/envs/tdhoang/lib/python3.12/site-packages/simglucose/params/vpatient_params.csv')
    patient_info = patient_info[patient_info['Name']==patient_id]
    
    fixed_values = {param_name : patient_info[param_name].values[0] for param_name in ['Gb', 'ka2', 'kd', 'kabs']}
    bw = float(patient_info.BW.values[0])
    u2ss = float(patient_info.u2ss.values[0])
    x0 = patient_info[['x0_ 4', 'x0_ 7', 'x0_ 1', 'x0_ 2', 'x0_ 3', 'x0_11', 'x0_12', 'x0_ 6', 'x0_ 4']].values[0].tolist()

    save_name = patient_id
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
        fixed_values=fixed_values,
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
        u2ss=u2ss,
        x0=x0
    )
    print(f"✅ Twinning completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReplayBG twinning and replay.")
    parser.add_argument("--method", type=str, default='map',
                        help="Twinning method to use: 'map' or 'mcmc'.")
    parser.add_argument("--test_data", type=str, default='./data/simulated/test_data.pt', help="Path to testing theta .pt file.")
    parser.add_argument("--save_folder", type=str, default='./', help="Path to save results.")

    args = parser.parse_args()
    main(args)
