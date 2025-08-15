

import os 
import sys
import argparse

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import seed_everything, metric_df_of_user_study
from data_loader.prompt_loader import load_yaml_config
from evaluation.utils.analysis_utils import corr_matrix_plot

#python 00_eval/user_study_7_corr_ours_ave.py --header wo_r_

if __name__ == "__main__":
    seed_everything(4321)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./eval_config.yaml"
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

    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    gen_method_list = config["gen_method_list"]
    prompt_type_list = config["prompt_type_list"]
    mode_list = config["mode_list"]
    results_dir = config["results_dir"]
    header = "w_r_" if args.reason else "wo_r_"


    # ----------------------
    METRIC_NAME = "GPT_ours"

    reg_dir = os.path.join(results_dir, METRIC_NAME + "_ave")
    os.makedirs(reg_dir, exist_ok=True)

    test_gpt_df = metric_df_of_user_study(
        gen_method_list, mode_list, prompt_type_list, config["results_dir"], METRIC_NAME, header=header)

    corr_matrix_plot(test_gpt_df, reg_dir)