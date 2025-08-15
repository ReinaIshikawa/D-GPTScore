
import os 
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
from evaluation.utils.user_study_utils import seed_everything, metric_df_all
from evaluation.utils.analysis_utils import radar_chart, evaluation_aspects
from data_loader.prompt_loader import load_yaml_config


if __name__ == "__main__":
    seed_everything(4321)
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
    parser.add_argument(
        '--gpt_model', 
        type=str, 
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini"]
    )
    parser.add_argument(
        '--gen-method', 
        type=str, 
        nargs='+',
        default=["01_CustomDiffusion", "02_OMG_lora", "03_OMG_instantID", "04_fastcomposer", "05_Mix-of-Show", "06_DreamBooth"] # add your method name here
    )
    parser.add_argument(
        '--type', 
        type=str,  
        nargs='+',
        default=["simple", "action+layout", "action+expression", "action+background", "all"]
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        nargs='+', 
        default=["easy", "medium", "hard"]
    )
    parser.add_argument(
        '--raderchart', 
        action="store_true", 
        default=False, 
    )
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    gen_method_list = args.gen_method
    prompt_type_list = args.type
    mode_list = args.mode
    results_dir = config["results_dir"]
    header = "w_r_" if args.reason else "wo_r_"

    # ----------------------
    if args.gpt_model == "gpt-4o":
        METRIC_NAME = "GPT_ours"
    elif args.gpt_model == "gpt-4o-mini":
        METRIC_NAME = "GPT4omini_ours"

    reg_dir = os.path.join(results_dir, METRIC_NAME + "_ave")
    os.makedirs(reg_dir, exist_ok=True)

    mean_list = []
    std_list = []
    benchmark_list = []
    print("-"*10)
    for gen_method in gen_method_list:
        print(gen_method)
        gpt_df = metric_df_all(
            gen_method_list=[gen_method], 
            mode_list=mode_list, 
            prompt_type_list=prompt_type_list, 
            results_dir=results_dir, 
            metric_name=METRIC_NAME, 
            header=header
        )
        decomposed_mean_df = gpt_df.mean(axis=0)
        decomposed_mean_list = np.array(decomposed_mean_df).flatten().tolist()
        mean_list.append(decomposed_mean_list)
        print("** Decomposed Score (1-5)**")
        for i in range(len(decomposed_mean_list)):
            print(f"{evaluation_aspects[f'score{i+1}']}: \t\t{decomposed_mean_list[i]:.3f}")
        gpt_df = (gpt_df-1) / (5-1) * (10-1) + 1
        benchmark_df = gpt_df.mean(axis=1).mean(axis=0)
        benchmark_list.append(benchmark_df.item())
        print("-"*3)
        print(f"** D-GPTScore (1-10)** \n{benchmark_df.item():.3f}")
        print("="*10)

    if args.raderchart:
        radar_chart(mean_list, reg_dir)