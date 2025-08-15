import subprocess
import sys
import os
import itertools

# please set the path to the OMG directory
# we added some modifications to the OMG directory to make it work with our code, especially for the saving of the generated images
GEN_SCRIPTS_DIR = os.path.expanduser("~/data/OMG")
PYTHON_EXECUTABLE = '/mnt/ssd2_4T/ishikawa/miniconda3/envs/OMG/bin/python'

sys.path.append(os.path.abspath(os.getcwd())) 
sys.path.append(GEN_SCRIPTS_DIR) 

env = os.environ.copy()
env['PYTHONPATH'] = f'{GEN_SCRIPTS_DIR}:' + env.get('PYTHONPATH', '')

from data_loader.prompt_loader import DataSaver, CCAlignBenchLoader, load_yaml_config

config = load_yaml_config(yaml_path="./gen_scripts/02_OMG_lora_prompt_config.yaml")

prompt_type_list = ["simple", "action+layout", "action+expression", "action+background", "all"]
mode_list = ["easy", "medium", "hard"]

csv_path = os.path.join(config["dir"],config["csv_file"])
dataloader = CCAlignBenchLoader(
    csv_path = csv_path,
    man_token = config["man_token"], 
    woman_token = config["woman_token"])

datasaver = DataSaver(prompt_type_list, mode_list, config)

#load options
neg_prompt = config["OPTION"]["neg_prompt"]
output_dir_path = config["OPTION"]["output_dir_path"]
model_path_man = config["OPTION"]["model_path_man"]
model_path_woman = config["OPTION"]["model_path_woman"]

for mode, prompt_type in itertools.product(mode_list, prompt_type_list):
    dir_name = os.path.join(output_dir_path, f"{mode}_{prompt_type}")
    os.makedirs(dir_name, exist_ok = True)

    index_list = range(dataloader.get_len_of_data(mode)) if config["index_list"] == None else config["index_list"]
    for idx in index_list:
        print("===============")
        print(f"mode:{mode} / prompt_type:{prompt_type}")
        data = dataloader.get_idx_info(mode, prompt_type, idx)
        id_ = data["id_"]
        p1_sex = data["p1_sex"]
        p2_sex = data["p2_sex"]
        p1_prompt = data["p1_prompt"]
        p2_prompt = data["p2_prompt"]
        prompt = data["prompt_class"]

        if mode=="easy":
            #================
            if p1_sex == "man":
                lora_path = model_path_man
            else:
                lora_path = model_path_woman
            rewrite = f"[{p1_prompt}]-*-[{neg_prompt}]"

            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/inference_lora.py', 
                '--prompt', prompt, 
                '--negative_prompt', neg_prompt,
                '--prompt_rewrite', rewrite,
                '--lora_path', lora_path,
                '--save_dir', dir_name,
                '--segment_type', 'GroundingDINO',
                '--prompt_id', f'{id_}',
                '--pt1', p1_sex
            ], capture_output=True, text=True, env=env)
            #================
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")

        else:
            #================
            rewrite = f"[{p1_prompt}]-*-[{neg_prompt}]|[{p2_prompt}]-*-[{neg_prompt}]"
            if p1_sex == "man":
                lora_path = f"{model_path_man}|{model_path_woman}"
            else:
                lora_path = f"{model_path_woman}|{model_path_man}"
            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/inference_lora.py', 
                '--prompt', prompt, 
                '--negative_prompt', neg_prompt,
                '--prompt_rewrite', rewrite,
                '--lora_path', lora_path,
                '--save_dir', dir_name,
                '--segment_type', 'GroundingDINO',
                '--prompt_id', f'{id_}',
                '--pt1', p1_sex,
                '--pt2', p2_sex
            ], capture_output=True, text=True, env=env)
            #================
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")

        datasaver.append_prompt(
            mode=mode, 
            prompt_type=prompt_type, 
            data={
                "id": int(id_),
                "prompt": prompt,
                "rewrite": rewrite
            }
        )
    datasaver.save_prompt(os.path.join(output_dir_path, "prompt.json"))

