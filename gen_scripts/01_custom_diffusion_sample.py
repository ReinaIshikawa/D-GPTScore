# accelerate launch gen_scripts/01_custom_diffusion_sample.py
import subprocess
import os
import itertools
import sys

GEN_SCRIPTS_DIR = os.path.expanduser("~/data/custom-diffusion")
PYTHON_EXECUTABLE = '/mnt/ssd2_4T/ishikawa/miniconda3/envs/ldm/bin/python'

sys.path.append(os.path.abspath(os.getcwd())) 
sys.path.append(GEN_SCRIPTS_DIR) 

from data_loader.prompt_loader import DataSaver, DataLoader, load_yaml_config

env = os.environ.copy()
env['PYTHONPATH'] = f'{GEN_SCRIPTS_DIR}:' + env.get('PYTHONPATH', '')

config = load_yaml_config(yaml_path="./gen_scripts/01_custom_diffusion_prompt_config.yaml")

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
# neg_prompt = config["OPTION"]["neg_prompt"]
output_dir_path = config["OPTION"]["output_dir_path"]
delta_ckpt_path_man = config["OPTION"]["delta_ckpt_path_man"]
delta_ckpt_path_woman = config["OPTION"]["delta_ckpt_path_woman"]
delta_ckpt_fused = config["OPTION"]["delta_ckpt_fused"]

for mode, prompt_type in itertools.product(mode_list, prompt_type_list):

    dir_name = os.path.join(output_dir_path, f"{mode}_{prompt_type}")
    os.makedirs(dir_name, exist_ok = True)

    index_list = range(dataloader.get_len_of_data(mode)) if config["index_list"] == None else config["index_list"]
    for idx in index_list:
        data = dataloader.get_idx_info(mode, prompt_type, idx)
        prompt = data["prompt_token"]
        id_ = data["id"]
        p1_sex = data["p1_sex"]
        p2_sex = data["p2_sex"]
        pt1 = data["pt1"]
        pt2 = data["pt2"]
        prompt_class = data["prompt_class"]

        if mode=="easy":
            delta_ckpt = delta_ckpt_path_man if p1_sex == "man" else delta_ckpt_path_woman
            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/src/diffusers_sample.py', 
                '--prompt', prompt, 
                '--delta_ckpt', delta_ckpt,
                '--ckpt', 'CompVis/stable-diffusion-v1-4',
                '--freeze_model', 'crossattn',
                '--batch_size', '1',
                '--outdir', dir_name,
                '--prompt_id', f'{id_}'
            ], capture_output=True, text=True, env=env)
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
        else:
            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/src/diffusers_sample.py', 
                '--prompt', prompt, 
                '--delta_ckpt', delta_ckpt_fused, 
                '--ckpt', 'CompVis/stable-diffusion-v1-4',
                '--freeze_model', 'crossattn',
                '--batch_size', '1',
                '--outdir', dir_name,
                '--prompt_id', f'{id_}'
            ], capture_output=True, text=True, env=env)
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")

        datasaver.append_prompt(
            mode=mode, 
            prompt_type=prompt_type, 
            data={
                "id": int(id_),
                "prompt": prompt,
            }
        )
    datasaver.save_prompt(os.path.join(output_dir_path, "prompt.json"))

