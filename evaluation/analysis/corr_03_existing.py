
import os 
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader, user_df_splitter, metric_df_of_user_study, seed_everything
from evaluation.utils.analysis_utils import calc_pearson_corr, calc_spearman_corr, calc_rank
from data_loader.prompt_loader import load_yaml_config
    
# python evaluation/analysis/corr_03_existing.py

if __name__ == "__main__":
    seed_everything(4321)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./evaluation/user_study/user_study_config.yaml"
    )
    parser.add_argument(
        '--corr', 
        type=str, 
        default="ave", 
        choices=["ave", "median"]
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
    # ----------------------

    usr_ave = False

    reg_dir = os.path.join(results_dir, "existing")
    os.makedirs(reg_dir, exist_ok=True)

    json_path = os.path.join(reg_dir, "corr.json")
    json_data = {}

    new_max = 10
    new_min = 1
    for metric_id, metric_name in enumerate(eval_metric_list):
        min_val, max_val = scale_list[metric_id]
        json_data[metric_name] = {}
        test_pred_list = []
        test_y_list = []
        corr_list = []
        for gen_method in gen_method_list:
            print(f"test:{gen_method}")
            
            ### =========================================================
            ### read user study result / culmun: ["01", "02", "03", ...]
            ### =========================================================
            user_df = user_study_reader(users, gen_method_list, user_study_data_dir)
            test_usr_df = user_df_splitter(
                [gen_method], user_df, mode_list, prompt_type_list, users)
            
            ### =====================
            ### load scores of existing evaluation methods
            ### =====================
            # ['score']
            test_pred = metric_df_of_user_study(
                [gen_method], 
                mode_list, 
                prompt_type_list, 
                results_dir, 
                metric_name, 
                header="")
            
            test_pred=test_pred.to_numpy().flatten()
            test_pred = (test_pred-min_val) / (max_val-min_val) * (new_max-new_min) + new_min

            ### =====================
            ### calculate correlation
            ### =====================  
            if args.corr=="ave":
                test_y=np.array(test_usr_df.mean(axis=1))
            if args.corr=="median":
                test_y=np.array(test_usr_df.median(axis=1))

            # calc pearson corr
            pearson_corr = calc_pearson_corr(test_pred, test_y)
            json_data[metric_name][f"pearson_corr_{gen_method}"] = pearson_corr

            test_pred_list.append(test_pred)
            test_y_list.append(test_y)

         # calc pearson corr
        test_pred_con = np.concatenate(test_pred_list)
        test_y_con = np.concatenate(test_y_list)
        pearson_corr = calc_pearson_corr(test_pred_con, test_y_con)
        print("pearson corr:", pearson_corr)
        json_data[metric_name]["pearson_corr"] = pearson_corr

        # calc spearman corr
        test_pred_rank = calc_rank(test_pred_list)
        test_y_rank = calc_rank(test_y_list)
        for pr, yr, gen in zip(test_pred_rank, test_y_rank, gen_method_list):
            spearman_corr = calc_spearman_corr(pr, yr)
            json_data[metric_name][f"spearman_corr_{gen}"] = spearman_corr

        test_pred_rank = np.concatenate(test_pred_rank)
        test_y_rank = np.concatenate(test_y_rank)
        spearman_corr = calc_spearman_corr(test_pred_rank, test_y_rank)
        json_data[metric_name]["spearman_corr"] = spearman_corr

    print(json_data)

    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    