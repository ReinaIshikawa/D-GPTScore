
import os 
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import seed_everything, metric_df_all
from evaluation.utils.analysis_utils import radar_chart
from data_loader.prompt_loader import load_yaml_config


if __name__ == "__main__":
    seed_everything(4321)
    seed_everything(4321)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./evaluation/user_study/user_study_config.yaml"
    )
    parser.add_argument(
        '--plus', 
        action="store_true", 
        default=False, 
    )
    parser.add_argument(
        '--corr', 
        type=str, 
        default="ave", 
        choices=["ave", "median"]
    )
    parser.add_argument(
        '--reason', 
        action="store_true", 
        default=False, 
    )
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    gen_method_list = config["gen_method_list"]
    prompt_type_list = config["prompt_type_list"]
    eval_metric_list = config["eval_metric_list"]
    mode_list = config["mode_list"]
    users = config["users"]
    scale_list = config["scale"]
    user_study_data_dir = os.path.join(
        config["results_dir"], 
        "user_study"
    )
    results_dir = config["results_dir"]
    header = "w_r_" if args.reason else "wo_r_"

    # ----------------------
    METRIC_NAME = "GPT_ours"

    reg_dir = os.path.join(results_dir, METRIC_NAME + "_ave")
    os.makedirs(reg_dir, exist_ok=True)

    mean_list = []
    std_list = []
    for gen_method in gen_method_list:
        gpt_df = metric_df_all(
            gen_method_list=[gen_method], 
            mode_list=mode_list, 
            prompt_type_list=prompt_type_list, 
            results_dir=results_dir, 
            metric_name=METRIC_NAME, 
            header=header
        )
        mean_df = gpt_df.mean(axis=0)
        mean_list.append(np.array(mean_df).flatten().tolist())
        std_df = gpt_df.std(axis=0)
        std_list.append(np.array(std_df).flatten().tolist())
    print(mean_list)
    print(std_list)

    radar_chart(mean_list, reg_dir)