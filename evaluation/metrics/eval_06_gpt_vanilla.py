import os
import json
import sys
import argparse
import itertools
import pandas as pd
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import DataLoader, load_yaml_config
from evaluation.utils.eval_utils import get_csv_path, get_gen_output_path
from evaluation.utils.user_study_utils import get_user_study_target
from evaluation.utils.gpt_utils import get_messages_vanilla
from openai import OpenAI
from pydantic import BaseModel

# python evaluation/metrics/eval_06_gpt_vanilla.py --part --gen-method xxx

client = OpenAI()

class Score_w_reason(BaseModel):
    Rationale:str
    Score:int
    
class Score_wo_reason(BaseModel):
    Score:int

METRIC_NAME = "GPT_vanilla"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./eval_config.yaml"
    )
    parser.add_argument(
        '--gen-method', 
        type=str, 
        default="",
        choices=["01_CustomDiffusion", "02_OMG_lora", "03_OMG_instantID", "04_fastcomposer", "05_Mix-of-Show", "06_DreamBooth"]
    )
    parser.add_argument(
        '--gpt_model', 
        type=str, 
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini"]
    )
    parser.add_argument(
        '--reason', 
        action='store_true',
        help="reason flag"
    )
    parser.add_argument(
        '--start', 
        type=int, 
        default=0,
        help="start index"
    )
    parser.add_argument(
        '--end', 
        type=int, 
        default=-1, 
        help="end index"
    )
    parser.add_argument(
        '--type', 
        type=str,
        nargs='+',
        default=[],
        choices=["simple", "action+layout", "action+expression", "action+background", "all"])
    parser.add_argument(
        '--mode', 
        type=str, 
        nargs='+',
        default=[],
        choices=["easy", "medium", "hard"]
    )
    parser.add_argument(
        '--part', 
        action='store_true', 
        help="execute only the part of user study"
    )

    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    if args.type != []:
        config["prompt_type_list"] = args.type
    if args.mode != []:
        config["mode_list"] = args.mode
    if args.gen_method != "":
        gen_method_list = [args.gen_method]
    else:
        gen_method_list = config["gen_method_list"]

    if args.reason:
        response_format = Score_w_reason
        header = "w_r_"
    else:
        response_format = Score_wo_reason
        header = "wo_r_"

    csv_path = os.path.join(config["dir"],config["csv_file"])
    bg_path = os.path.join(config["dir"],config["bg_file"])
    dataloader = DataLoader(
        csv_path = csv_path,
        bg_path = bg_path,
        surrounings_type = config["surrounings_type"], 
        man_token = config["man_token"], 
        woman_token = config["woman_token"], 
        debug = config["debug"]
    )

    
    for gen_method in gen_method_list:
        output_csv_path = get_csv_path(
            results_dir=config["results_dir"], 
            metric_name=METRIC_NAME, 
            gen_method=gen_method,
            header=header
        )

        if args.part:
            target_list = get_user_study_target()
        else:
            target_list = []
            for mode, prompt_type in itertools.product(config["mode_list"], config["prompt_type_list"]):
                # get index list
                if config["index_list"] == None:
                    if args.end != -1:
                        index_list = range(args.start, min(args.end+1, dataloader.get_len_of_data(mode)))
                    else:
                        index_list = range(args.start, dataloader.get_len_of_data(mode))
                else:
                    index_list = config["index_list"]
                target_list.append((mode, prompt_type, idx) for idx in index_list)
            
        # ===============================
        # for each generated image
        # ===============================
        for mode, prompt_type, idx in target_list:
            print("===============")
            print(f"mode:{mode} / prompt_type:{prompt_type} / id_:{idx}")

            prompt_info = dataloader.get_idx_info(mode, prompt_type, idx)
            id_ = prompt_info["id_"]
            prompt_token = prompt_info["prompt_token"]
            p1_sex = prompt_info["p1_sex"]
            p2_sex = prompt_info["p2_sex"]


            # get generated image path
            generated_img_path = get_gen_output_path(
                config["gen_output_dir"], 
                gen_method, 
                mode, 
                prompt_type, 
                id_
            )

            # get reference image path
            if mode == "easy":
                if p1_sex == "man":
                    ref_image_path1 = config["man_ref_image"]
                else:
                    ref_image_path1 = config["woman_ref_image"]
                ref_image_path2 = None
            else:
                if p1_sex == "man":
                    ref_image_path1 = config["man_ref_image"]
                    ref_image_path2 = config["woman_ref_image"]
                else:
                    ref_image_path1 = config["woman_ref_image"]
                    ref_image_path2 = config["man_ref_image"]

                    
            messages = get_messages_vanilla(
                prompt_token = prompt_token,
                ref_image_path1 = ref_image_path1,
                ref_image_path2 = ref_image_path2,
                generated_img_path = generated_img_path,
                reason_flag=args.reason
            )

            # get response from gpt
            response = client.beta.chat.completions.parse(
                model= args.gpt_model,
                temperature= 0.0,
                seed=1234,
                messages= messages, # type: ignore
                response_format= response_format
            )

            content = response.choices[0].message.content
            parsed_content = json.loads(content) # type: ignore

            # save output
            gpt_output = {}
            gpt_output["mode"]=mode
            gpt_output["prompt_type"]=prompt_type
            gpt_output["id_"]=id_
            gpt_output[f"score"] = parsed_content[f"Score"]
            if args.reason:
                gpt_output[f"reason"] = parsed_content[f"Rationale"]
                print("Score: {}\nReason: {}".format(parsed_content[f"Score"], parsed_content[f"Rationale"]))
            else:
                gpt_output[f"reason"] = ""
                print("Score: {}".format(parsed_content[f"Score"]))
            gpt_output[f"comp_tok_sum"]=response.usage.completion_tokens # type: ignore
            gpt_output[f"prompt_tok_sum"]=response.usage.prompt_tokens # type: ignore

            df = pd.DataFrame(gpt_output, index=[0]) # type: ignore
            df = df[
                ["mode", "prompt_type","id_"] \
                + ["score"] \
                + ["reason"] \
                + ["comp_tok_sum", "prompt_tok_sum"]
            ]

            file_exists = os.path.isfile(output_csv_path)
            with open(output_csv_path, mode='a', newline='') as file:
                df.to_csv(file, header=not file_exists, index=False)