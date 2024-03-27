import yaml
import os
from src.data_loader import load_and_merge_data
from src.feature_engineering import generate_confounders
from src.causal_model import PriceElasticityModel

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    data_dir = config['data']['raw_dir']
    
    print("Loading and merging M5 dataset...")
    df = load_and_merge_data(data_dir)
    
    print("Engineering confounders and features...")
    df_feat = generate_confounders(df)
    
    # Define arrays for causal estimation
    # Y = sales, T = log_price, W = confounders, X = effect modifiers
    print("Initializing and fitting Causal Forest DML model...")
    model = PriceElasticityModel(config)
    # model.fit(Y, T, X, W)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()