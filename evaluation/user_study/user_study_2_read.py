import os
import sys

from sklearn.base import config_context
sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader
import argparse
from data_loader.prompt_loader import load_yaml_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default="./evaluation/user_study/user_study_config.yaml")
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    gen_method_list = config["gen_method_list"]
    users = config["users"]
    user_study_data_dir = os.path.join(
        config["results_dir"], 
        "user_study"
    )
    save_path = os.path.join(user_study_data_dir, "total", 'user_study_result.csv')

    # ----------------------
    df = user_study_reader(
        users, gen_method_list, user_study_data_dir)
    df.to_csv(save_path, index=False)
    print(f"User study data saved to {save_path}")