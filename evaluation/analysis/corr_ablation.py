
import os 
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader, user_df_splitter, metric_df_of_user_study, seed_everything
from data_loader.prompt_loader import load_yaml_config
from evaluation.utils.analysis_utils import ablation_bar_plot
from evaluation.utils.analysis_utils import calc_pearson_corr, calc_spearman_corr, calc_rank

#python evaluation/analysis/corr_ablation.py --reason


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
    json_path = os.path.join(reg_dir, "corr_ablation.json")
    json_data = {}

    for ea in range(0,19):
        if ea==0:
            ablation_target = "default"
        else:
            ablation_target = f"score{ea}"
        print(f"ablation_target:{ablation_target}")
        
        json_data[ablation_target] = {}
        test_pred_list = []
        test_y_list = []
        for i, gen_method in enumerate(gen_method_list):
            train_gen_method_list = gen_method_list.copy()
            test_gen_method = train_gen_method_list.pop(i)
            
            
            ### =========================================================
            ### read user study result / culmun: ["01", "02", "03", ...]
            ### =========================================================
            user_df = user_study_reader(users, gen_method_list, user_study_data_dir)
            test_usr_df = user_df_splitter(
                [gen_method], user_df, mode_list, prompt_type_list, users)

            ### =========================================================
            ### read gpt result / culmun:  ["score1", "score2", ..., "score18"]
            ### =========================================================
            test_gpt_df = metric_df_of_user_study(
                [gen_method], mode_list, prompt_type_list, config["results_dir"], METRIC_NAME, header=header)

            new_min, new_max = 1, 10
            test_gpt_df = (test_gpt_df-1) / (5-1) * (new_max-new_min) + new_min
            if ea!=0:
                test_gpt_df = test_gpt_df.drop(columns=[ablation_target])
            
            # =======================
            if args.corr=="ave":
                test_y=np.array(test_usr_df.mean(axis=1))
            if args.corr=="median":
                test_y=np.array(test_usr_df.median(axis=1))
            
            test_pred = np.array(test_gpt_df.mean(axis=1))
 
            test_corr = calc_pearson_corr(test_pred, test_y)
            test_pred_list.append(test_pred)
            test_y_list.append(test_y)


        # calc pearson corr
        test_pred_con = np.concatenate(test_pred_list)
        test_y_con = np.concatenate(test_y_list)
        pearson_corr = calc_pearson_corr(test_pred_con, test_y_con)
        print("total pearson corr:", pearson_corr)
        json_data[ablation_target]["total_pearson_corr"] = pearson_corr

        # calc spearman corr
        test_pred_rank = calc_rank(test_pred_list)
        test_y_rank = calc_rank(test_y_list)
        test_pred_rank = np.concatenate(test_pred_rank)
        test_y_rank = np.concatenate(test_y_rank)
        spearman_corr = calc_spearman_corr(test_pred_rank, test_y_rank)
        print("total spearman corr:", spearman_corr)
        json_data[ablation_target]["total_spearman_corr"] = spearman_corr

    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    ablation_bar_plot(json_data, reg_dir)