
import os 
import sys
import argparse

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import user_study_reader, user_score_target_reader, seed_everything, metric_target_reader
from data_loader.prompt_loader import load_yaml_config
    
# python evaluation/analysis/_check_score.py --type all --mode hard --idx 81 
if __name__ == "__main__":
    seed_everything(4321)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./eval_config.yaml"
    )
    parser.add_argument(
        '--type', 
        type=str, 
        choices=["simple", "action+layout", "action+expression", "action+background", "all"]
    )
    parser.add_argument(
        '--mode', 
        type=str,
        choices=["easy", "medium", "hard"]
    )
    parser.add_argument(
        '--reason', 
        action="store_true", 
        default=False, 
    )
    parser.add_argument('--idx', type=int, default=-1)
    args = parser.parse_args()

    config = load_yaml_config(yaml_path=args.yaml_path)
    # target
    prompt_type = args.type
    mode = args.mode
    idx = args.idx
    

    gen_method_list = config["gen_method_list"]
    eval_metric_list = config["eval_metric_list"]
    prompt_type_list = config["prompt_type_list"]
    mode_list = config["mode_list"]
    users = config["users"]
    results_dir = config["results_dir"]
    scale_list = config["scale"]
    user_study_data_dir = os.path.join(
        config["results_dir"], 
        "user_study"
    )
    header = "w_r_" if args.reason else "wo_r_"
    # ----------------------

    METRIC_NAME = "GPT_ours"

    for gen_method in gen_method_list:
        print(f"====== gen_method: {gen_method} ======")
        gpt_score = metric_target_reader(
            gen_method, 
            METRIC_NAME, 
            results_dir, 
            mode, 
            prompt_type, 
            idx, 
            header=header
        )
        # user score
        user_df = user_study_reader(
            users, 
            gen_method_list, 
            user_study_data_dir
        )
        user_score = user_score_target_reader(
            gen_method, 
            user_df, 
            users, 
            mode, 
            prompt_type, 
            idx
        )
        print(f"user score: {user_score.mean(axis=1)}")

        # gpt score
        print(f"gpt score:")
        for i in range(gpt_score.shape[1]):
            print(f"gpt score {i}: {gpt_score.iloc[0,i].item()}") # type: ignore
        new_min, new_max = 1, 10
        gpt_score = (gpt_score-1) / (5-1) * (new_max-new_min) + new_min
        print(f"gpt total score: {gpt_score.mean(axis=1).item()}")

        for eval_metric, scale in zip(eval_metric_list, scale_list):
            current_score = metric_target_reader(
                gen_method, 
                eval_metric, 
                results_dir, 
                mode, 
                prompt_type, 
                idx)

            print(f"{eval_metric} score: {current_score}")


    