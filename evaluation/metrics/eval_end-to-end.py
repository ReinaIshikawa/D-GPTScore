import os
import json
import sys
import argparse
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import load_yaml_config
from evaluation.utils.gpt_utils import get_messages_part
from openai import OpenAI
from pydantic import BaseModel

# python evaluation/metrics/eval_end-to-end.py \
# --generated_img_path ./results/eval_end-to-end/generated_img.png \
# --concept_path ./CC-AlignBench/man_1/image/0.jpeg ./CC-AlignBench/woman_1/image/0.jpeg \
# --prompt "a man and a woman, shaking hands" \

client = OpenAI()

class Score_w_reason(BaseModel):
    Rationale:str
    Score:int

class Score_wo_reason(BaseModel):
    Score:int



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
        '--prompt', 
        type=str, 
        default="",
        help="prompt used to generate the input image"
    )
    parser.add_argument(
        '--generated_img_path', 
        type=str, 
        default="",
        help="path to the image you want to evaluate"
    )
    parser.add_argument(
        '--concept_path', 
        type=str, 
        nargs='*',
        default=["./CC-AlignBench/man_1/image/0.jpeg", "./CC-AlignBench/woman_1/image/0.jpeg"],
        help="path to concept images"
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default="results/eval_end-to-end",
        help="output image path"
    )
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)

    # ===============================
    # get args
    # ===============================

    if args.reason:
        response_format = Score_w_reason
        header = "w_r_"
    else:
        response_format = Score_wo_reason
        header = "wo_r_"

    if args.generated_img_path == "":
        raise ValueError("Please indicate the generated image path with --generated_img_path")
    if not os.path.exists(args.generated_img_path):
        raise ValueError(f"Generated image path {args.generated_img_path} does not exist. Please check the path.")
    generated_img_path = args.generated_img_path

    if len(args.concept_path) < 1 or len(args.concept_path) > 2:
        raise ValueError("Please indicate 1 or 2 concept image paths with --concept_path")

    # check if the concept image paths exist
    if not os.path.exists(args.concept_path[0]):
        raise ValueError(f"Concept image path {args.concept_path[0]} does not exist. Please check the path.")

    # set ref_image_path1
    ref_image_path1 = args.concept_path[0]      

    # set ref_image_path2
    if len(args.concept_path) == 2:
        if not os.path.exists(args.concept_path[1]):
            raise ValueError(f"Concept image path {args.concept_path[1]} does not exist. Please check the path.")
        ref_image_path2 = args.concept_path[1]
    else:
        ref_image_path2 = None


    if args.prompt == "":
        raise ValueError("Please indicate the prompt with --prompt")

    
    # initialize output
    gpt_output = {}

    # ===============================
    # for each EA(Evaluation Aspect)
    # ===============================
    for ea_id in range(1,19):
        print(f"- EA{ea_id}")

        # get messages for gpt
        messages = get_messages_part(
            prompt_token = args.prompt,
            ref_image_path1 = ref_image_path1,
            ref_image_path2 = ref_image_path2,
            generated_img_path = generated_img_path,
            ea_id = ea_id,
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
        gpt_output[f"score{ea_id}"] = parsed_content["Score"]
        if args.reason:
            gpt_output[f"reason{ea_id}"] = parsed_content["Rationale"]
            print("Score: {}\nReason: {}".format(
                parsed_content["Score"], parsed_content["Rationale"]))
        else:
            gpt_output[f"reason{ea_id}"] = ""
            print("Score: {}".format(
                parsed_content["Score"]))

        gpt_output[f"comp_tokens{ea_id}"]=response.usage.completion_tokens # type: ignore
        gpt_output[f"prompt_tokens{ea_id}"]=response.usage.prompt_tokens # type: ignore

    print(gpt_output)
    # save output
    with open(args.output_path, "w") as f:
        json.dump(gpt_output, f, indent=4)

    print(f"Output saved to {args.output_path}")
