# https://github.com/huggingface/diffusers/tree/main/examples/research_projects/multi_subject_dreambooth
# _run_training.sh -> models
import sys
import os
import itertools

sys.path.append(os.path.abspath(os.getcwd())) 

from data_loader.prompt_loader import DataSaver, DataLoader, load_yaml_config

from diffusers import StableDiffusionPipeline
import torch

config = load_yaml_config(yaml_path="./gen_scripts/06_DreamBooth_prompt_config.yaml")

prompt_type_list = ["simple", "action+layout", "action+expression", "action+background", "all"]
mode_list = ["easy", "medium", "hard"]

csv_path = os.path.join(config["dir"],config["csv_file"])
bg_path = os.path.join(config["dir"],config["bg_file"])
dataloader = DataLoader(
    csv_path = csv_path,
    bg_path = bg_path,
    surrounings_type = config["surrounings_type"], 
    man_token = config["man_token"], 
    woman_token = config["woman_token"], 
    debug = config["debug"])

datasaver = DataSaver(prompt_type_list, mode_list, config)

#load options
neg_prompt = config["OPTION"]["neg_prompt"]
output_dir_path = config["OPTION"]["output_dir_path"]

model_id = config["OPTION"]["model_path"]
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,safety_checker=None).to("cuda")

for mode, prompt_type in itertools.product(mode_list, prompt_type_list):

    dir_name = os.path.join(output_dir_path, f"{mode}_{prompt_type}")
    os.makedirs(dir_name, exist_ok = True)

    index_list = range(dataloader.get_len_of_data(mode)) if config["index_list"] == None else config["index_list"]

    for idx in index_list:
        data = dataloader.get_idx_info(mode, prompt_type, idx)
        id_ = data["id"]
        prompt_token = data["prompt_token"]
        pt1 = data["pt1"]
        pt2 = data["pt2"]
        p1_sex = data["p1_sex"]
        p2_sex = data["p2_sex"]
        prompt = prompt_token
        
        image = pipe(prompt, num_inference_steps=200, guidance_scale=7.5).images[0]
        savepath = os.path.join(dir_name, f"{id_:03}.png")
        image.save(savepath)

        datasaver.append_prompt(
            mode=mode, 
            prompt_type=prompt_type, 
            data={
                "id": int(id_),
                "prompt": prompt,
            }
        )
    datasaver.save_prompt(os.path.join(output_dir_path, "prompt.json"))


