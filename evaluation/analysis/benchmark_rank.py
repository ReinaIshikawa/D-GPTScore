
import os 
import sys
import argparse
import numpy as np


sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader, user_df_splitter, metric_df_of_user_study, seed_everything
from evaluation.utils.analysis_utils import calc_rank, rank_plot
from data_loader.prompt_loader import load_yaml_config

# python evaluation/analysis/benchmark_rank.py 

if __name__ == "__main__":
    seed_everything(4321)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./eval_config.yaml"
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

    reg_dir = os.path.join(results_dir, METRIC_NAME)
    os.makedirs(reg_dir, exist_ok=True)

    user_df = user_study_reader(users, gen_method_list, user_study_data_dir)

    json_data = {}
    json_data["Ours"]=[]
    json_data["UserPreference"]=[]
    for eval_metric in eval_metric_list:
        json_data[eval_metric]=[]


    for i in range(len(gen_method_list)):
        test_gen_method = gen_method_list[i]
        print(f"test:{test_gen_method}")

        # current method result  ["score"]
        for eval_metric, scale in zip(eval_metric_list, scale_list):
            test_existing_df = metric_df_of_user_study(
                [test_gen_method], mode_list, prompt_type_list, results_dir, eval_metric)
            json_data[eval_metric].append(
                np.array(test_existing_df["score"]).flatten()
            )
        
        ### =========================================================
        ### read user study result / culmun: ["01", "02", "03", ...]
        ### =========================================================
        test_usr_df = user_df_splitter(
            [test_gen_method], user_df, mode_list, prompt_type_list, users)
        ### =========================================================
        ### read gpt result / culmun:  ["score1", "score2", ..., "score18"]
        ### =========================================================
        test_gpt_df = metric_df_of_user_study(
            [test_gen_method], mode_list, prompt_type_list, config["results_dir"], METRIC_NAME, header=header)
        
        json_data["UserPreference"].append(
            np.array(test_usr_df.mean(axis=1)).flatten())
        json_data["Ours"].append(
            np.array(test_gpt_df.mean(axis=1)).flatten())

    user_rank = calc_rank(json_data["UserPreference"])
    ours_rank = calc_rank(json_data["Ours"])

    user_rank_ave = []
    for i in range(len(user_rank)):
        user_rank_ave.append(user_rank[i].mean())
    print("user_rank_ave", user_rank_ave)

    ours_rank_ave = []
    for i in range(len(ours_rank)):
        ours_rank_ave.append(ours_rank[i].mean())
    print("ours_rank_ave", ours_rank_ave)


    eval_metric_rank_ave_list = []
    for eval_metric in eval_metric_list:
        eval_metric_rank = calc_rank(json_data[eval_metric])
        eval_metric_rank_ave = []
        for i in range(len(eval_metric_rank)):
            eval_metric_rank_ave.append(eval_metric_rank[i].mean())
        eval_metric_rank_ave_list.append(eval_metric_rank_ave)
        print(f"{eval_metric} rank_ave", eval_metric_rank_ave)


    rank_list =  eval_metric_rank_ave_list + [ours_rank_ave] + [user_rank_ave]
    rank_plot(rank_list, reg_dir, eval_metric_list)
    


    