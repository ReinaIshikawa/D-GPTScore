import os 
import sys
import argparse
sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import gpt_df_tokennum_reader, seed_everything
from data_loader.prompt_loader import load_yaml_config

# python evaluation/analysis/check_gpt_tokennum.py 
# our result:
# comp_tok_sum         96.000000
# prompt_tok_sum    12090.867347

if __name__ == "__main__":
    seed_everything(4321)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./eval_config.yaml"
    )
    parser.add_argument(
        '--reason', 
        action="store_true", 
        default=False, 
    )
    parser.add_argument(
        '--gpt_model', 
        type=str, 
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "vanilla"]
    )
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    gen_method_list = config["gen_method_list"]
    prompt_type_list = config["prompt_type_list"]
    mode_list = config["mode_list"]
    users = config["users"]
    results_dir = config["results_dir"]

    header = "w_r_" if args.reason else "wo_r_"
    # header = ""

    # ----------------------
    if args.gpt_model == "gpt-4o":
        METRIC_NAME = "GPT_ours"
    elif args.gpt_model == "gpt-4o-mini":
        METRIC_NAME = "GPT4omini_ours"
    elif args.gpt_model == "vanilla":
        METRIC_NAME = "GPT_vanilla"

    reg_dir = os.path.join(results_dir, METRIC_NAME)
    os.makedirs(reg_dir, exist_ok=True)
    
    json_path = os.path.join(reg_dir, "gpt_tokennum.json")
    json_data = {}
    mean_list = []
    std_list = []
    gpt_df = gpt_df_tokennum_reader(
            gen_method_list=gen_method_list, 
            gpt_method=METRIC_NAME,
            results_dir=results_dir, 
            header=header
    )
    print(gpt_df)