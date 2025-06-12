
# Simulation-Based Inference for T1D Digital Twin

This repository implements a simulation-based inference (SBI) pipeline for building a digital twin of individuals with Type 1 Diabetes (T1D).

---

## 🛠️ Setup Environment

```bash
conda env create -f environment.yml
conda activate sbit1d
cd py_replay_bg
pip install -e .
```

## 📁 Directory Structure

- `data/`: Contains both raw and simulated data  
  - `simulated/`: Simulated training and testing data  
    - `precomputed_train_data.pt`: 5000 training samples (theta_hat, CGM)  
    - `precomputed_test_data.pt`: 50 test samples (theta_hat, CGM)  
    - `data_day_1_{0..49}_extended.pt`: Simulated CGM for extended setting (48h)  (setting 2)
    - `data_day_1_{0..49}_noise.pt`: Simulated CGM for altered meal setting  (setting 3)
  - `data_day_1.csv`: 24h CHO, insulin, glucose, timestamp data (from `py_replay_bg`)  
  - `data_day_1_extended.csv`: 48h version of the data (from `py_replay_bg`)  
  - `patient_info.csv`: Metadata for patients (from `py_replay_bg`)  
  - `noise_meal.pt`: 50 altered meal scenarios for setting 3
- `plots/`: Directory to save plots
- `py_replay_bg/`: Python implementation of `replaybg` with added SBI functionality
- `results/`: Outputs from baseline inference methods  
  - `map/`: MAP parameter estimates  
  - `mcmc/`: MCMC parameter estimates  
  - `replay/`: Simulated CGM using estimated parameters  
  - `workspace/`: Simulated CGM returned by `replaybg`
- `sbi_results/`: Parameter estimates and CGM simulations from SBI
- `trained_models/`: Trained SBI models  
  - e.g., `pretrained_density_estimator.pt` used in the paper


## 🚀 How to Run

### 1. Generate Training and Testing Data

```bash
python gen_data_sbi.py --num_train 5000 --num_test 50 --save_path ./data/simulated 
```
This will generate training and testing data saved at `./data/simulated/train_data.pt`, `./data/simulated/test_data.pt`, respectively. Alternatively, use the precomputed files provided: `./data/simulated/precomputed_train_data.pt` and `./data/simulated/precomputed_test_data.pt`
    

----------

### 2. Run Baseline Estimation (MAP or MCMC)

```bash
python baseline.py --idx {idx} --method {method} --test_data {test_data}
```
-   `idx`: Integer from `0` to `49` (index of test sample)
-   `method`: One of `map` or `mcmc`
-   `test_data`: Path to test dataset (`./data/simulated/test_data.pt` or precomputed file)
    

----------

### 3. Train SBI Model

```bash
python train_sbi.py --train_data {train_data} --output_path ./trained_models/density_estimator.pt
```
-   `train_data`: Path to training dataset (`./data/simulated/train_data.pt` or precomputed file)
-   `output_path`: Path to save the trained model . A pretrained model is also available at `./trained_models/pretrained_density_estimator.pt`
    

----------

### 4. Evaluation

Open and run the notebook `evaluation.ipynb` 

This notebook allows to re-produce tables and figures in the paper.
    

----------
