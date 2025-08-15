import os 
import sys
import argparse
sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import seed_everything, metric_df_all
from data_loader.prompt_loader import load_yaml_config

if __name__ == "__main__":
    seed_everything(4321)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./eval_config.yaml"
    )
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    gen_method_list = config["gen_method_list"]
    prompt_type_list = config["prompt_type_list"]
    mode_list = config["mode_list"]
    users = config["users"]
    results_dir = config["results_dir"]
    eval_metric_list = config["eval_metric_list"]

    # ----------------------
    METRIC_NAME = "GPT_ours"

    for eval_metric in eval_metric_list:
        current_df = metric_df_all(
            gen_method_list=gen_method_list, 
            mode_list=mode_list, 
            prompt_type_list=prompt_type_list, 
            results_dir=results_dir, 
            metric_name=eval_metric, 
            header=""
        )

        min_values = current_df.min(axis=0)
        max_values = current_df.max(axis=0)
        mean_values =current_df.mean(axis=0)
        std_values = current_df.std(axis=0)

        print(f"====== {eval_metric} ======")
        print(f"min_values: {min_values}")
        print(f"max_values: {max_values}")
        print(f"mean_values: {mean_values}")
        print(f"std_values: {std_values}")

