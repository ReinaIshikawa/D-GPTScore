import os 
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader, user_df_splitter, metric_df_of_user_study, metric_df_all, seed_everything
from data_loader.prompt_loader import load_yaml_config
from evaluation.utils.analysis_utils import comparison_plot
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# python evaluation/analysis/score_comparison.py

if __name__ == "__main__":
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
    parser.add_argument(
        '--mode', 
        type=str, 
        nargs='+', 
        default=[],
        choices=["easy", "medium", "hard"]
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

    if args.mode != []:
        mode_list = args.mode

    # ----------------------
    METRIC_NAME = "GPT_ours"

    reg_dir = os.path.join(results_dir, METRIC_NAME)
    os.makedirs(reg_dir, exist_ok=True)

    ### =========================================================
    ### read user study result / culmun: ["01", "02", "03", ...]
    ### =========================================================
    user_df = user_study_reader(users, gen_method_list, user_study_data_dir)
    test_usr_df = user_df_splitter(
        gen_method_list, user_df, mode_list, prompt_type_list, users)

    ### =========================================================
    ### read gpt result / culmun:  ["score1", "score2", ..., "score18"]
    ### =========================================================
    
    test_gpt_df = metric_df_of_user_study(
        gen_method_list, mode_list, prompt_type_list, config["results_dir"], METRIC_NAME, header=header)
    new_min, new_max = 1, 10
    test_gpt_df = (test_gpt_df-1) / (5-1) * (new_max-new_min) + new_min
    
    ### =========================================================
    ### linear regression
    ### =========================================================
    if args.corr=="ave":
        train_y=np.array(test_usr_df.mean(axis=1))
    if args.corr=="median":
        train_y=np.array(test_usr_df.median(axis=1))

    train_x=test_gpt_df.to_numpy()
    
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

    linear_test_pred_list = []
    average_test_pred_list = []
    for gen_method in gen_method_list:

        all_gpt_df = metric_df_all(
            [gen_method], 
            mode_list, 
            prompt_type_list, 
            config["results_dir"], 
            METRIC_NAME, 
            header=header
        )
        # linear regression
        test_x=all_gpt_df.to_numpy()
        linear_test_pred = model.predict(test_x)
        linear_test_pred = linear_test_pred.mean()
        linear_test_pred_list.append(linear_test_pred)
        print(f"linear test_pred of {gen_method}: {linear_test_pred}")

        # average score
        new_min, new_max = 1, 10
        normalized_x = (all_gpt_df-1) / (5-1) * (new_max-new_min) + new_min
        test_pred = normalized_x.mean(axis=1).to_numpy()
        average_test_pred = test_pred.mean()
        average_test_pred_list.append(average_test_pred)
        print(f"average test_pred of {gen_method}: {average_test_pred}")

    comparison_plot(
        linear_test_pred_list, 
        average_test_pred_list, 
        reg_dir
    )



    