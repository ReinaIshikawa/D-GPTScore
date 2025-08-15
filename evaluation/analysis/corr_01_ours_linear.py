
import os 
import sys
import json
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import pandas as pd

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader, user_df_splitter, metric_df_of_user_study, seed_everything
from evaluation.utils.analysis_utils import calc_pearson_corr, calc_spearman_corr, calc_rank, scatter_plot, plot_feature_importance
from data_loader.prompt_loader import load_yaml_config

# python evaluation/analysis/corr_01_ours_linear.py
# python evaluation/analysis/corr_01_ours_linear.py --plus

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
        '--reg', 
        type=str, 
        default="", 
        choices=["ridge", "lasso", "elastic"]
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

    reg_dir = os.path.join(results_dir, METRIC_NAME + "_linear")
    if args.plus:
        reg_dir = reg_dir + "++"
    os.makedirs(reg_dir, exist_ok=True)

    json_path = os.path.join(reg_dir, "corr.json")
    json_data = {}

    test_pred_list = []
    test_y_list = []
    for i, gen_method in enumerate(gen_method_list):
        print(f"test:{gen_method}")
        train_gen_method_list = gen_method_list.copy()
        test_gen_method = train_gen_method_list.pop(i)
        json_data[test_gen_method]={}

        ### =========================================================
        ### read user study result / culmun: ["01", "02", "03", ...]
        ### =========================================================
        user_df = user_study_reader(users, gen_method_list, user_study_data_dir)
        train_usr_df = user_df_splitter(
            train_gen_method_list, user_df, mode_list, prompt_type_list, users)
        test_usr_df = user_df_splitter(
            [test_gen_method], user_df, mode_list, prompt_type_list, users)
        
        ### =========================================================
        ### read gpt result / culmun:  ["score1", "score2", ..., "score18"]
        ### =========================================================
        train_gpt_df = metric_df_of_user_study(
            train_gen_method_list, mode_list, prompt_type_list, config["results_dir"], METRIC_NAME, header=header)
        test_gpt_df = metric_df_of_user_study(
            [test_gen_method], mode_list, prompt_type_list, config["results_dir"], METRIC_NAME, header=header)

        ### =========================================================
        ### read exsiting metric result / culmun:  ["eval", "score2", ..., "score5"]
        ### =========================================================
        if args.plus:
            new_min, new_max = 1, 10
            train_combined = (train_gpt_df-1) / (5-1) * (new_max-new_min) + new_min
            test_combined = (test_gpt_df-1) / (5-1) * (new_max-new_min) + new_min


            for eval_metric, scale in zip(eval_metric_list, scale_list):
                min_val, max_val = scale
                # ['score']
                train_existing_df= metric_df_of_user_study(
                    train_gen_method_list, mode_list, prompt_type_list, config["results_dir"], eval_metric)

                test_existing_df = metric_df_of_user_study(
                    [test_gen_method], mode_list, prompt_type_list, config["results_dir"], eval_metric)

                train_existing_df = (train_existing_df-min_val) / (max_val-min_val) * (new_max-new_min) + new_min
                test_existing_df = (test_existing_df-min_val) / (max_val-min_val) * (new_max-new_min) + new_min

                train_combined = pd.concat([train_combined, train_existing_df], axis=1)
                test_combined = pd.concat([test_combined, test_existing_df], axis=1)

        else:
            train_combined = train_gpt_df
            test_combined = test_gpt_df

        ### =========================================================
        ### if corr: ave or median 
        ### =========================================================
        if args.corr=="ave":
            train_y=np.array(train_usr_df.mean(axis=1))
            test_y=np.array(test_usr_df.mean(axis=1))
        if args.corr=="median":
            train_y=np.array(train_usr_df.median(axis=1))
            test_y=np.array(test_usr_df.median(axis=1))

        train_x=np.array(train_combined)
        test_x =np.array(test_combined)
        
        # model selection
        if args.reg == "ridge":
            model = Ridge(alpha=100) # l1
        elif args.reg == "lasso":
            model = Lasso(alpha=0.1)
        elif args.reg == "elastic":
            model = ElasticNet(alpha=0.01, l1_ratio=0.5)
        else:
            model = LinearRegression()
        
        # fitting
        model.fit(train_x, train_y)

        # get feature imoprtance
        feature_importance = model.coef_
        print("Feature Importances:")
        print(feature_importance)

        if args.plus:
            plot_feature_importance(
                reg_dir, 
                feature_importance.tolist(), 
                f"importance_{test_gen_method}.pdf",
                True,
                existing_methods=eval_metric_list)
        else:
            plot_feature_importance(
                reg_dir, 
                feature_importance.tolist(), 
                f"importance_{test_gen_method}.pdf",
                True)

        # check correlation with test data
        test_pred = model.predict(test_x)
        test_corr = calc_pearson_corr(test_pred, test_y)

        test_pred_list.append(test_pred)
        test_y_list.append(test_y)
        json_data[test_gen_method]["pearson_corr"] = test_corr

    # calc pearson corr
    test_pred_con = np.concatenate(test_pred_list)
    test_y_con = np.concatenate(test_y_list)
    pearson_corr = calc_pearson_corr(test_pred_con, test_y_con)
    print("total pearson corr:", pearson_corr)
    json_data["total_pearson_corr"] = pearson_corr

    # calc spearman corr
    test_pred_rank = calc_rank(test_pred_list)
    test_y_rank = calc_rank(test_y_list)
    for pr, yr, gen in zip(test_pred_rank, test_y_rank, gen_method_list):
        spearman_corr = calc_spearman_corr(pr, yr)
        json_data[gen][f"spearman_corr"] = spearman_corr
    test_pred_rank = np.concatenate(test_pred_rank)
    test_y_rank = np.concatenate(test_y_rank)
    spearman_corr = calc_spearman_corr(test_pred_rank, test_y_rank)
    json_data["total_spearman_corr"] = spearman_corr

    scatter_plot(test_pred_con, test_y_con, gen_method_list, reg_dir,  f"plot.pdf")

    print(json_data)

    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
