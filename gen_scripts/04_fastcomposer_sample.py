# cuda:11.8
# conda activate fastcomposer
# nohup accelerate launch --mixed_precision=bf16 ./gen_scripts/04_fastcomposer_sample.py &
import subprocess
import sys
import os
import itertools

GEN_SCRIPTS_DIR = os.path.expanduser("~/data/fastcomposer")
PYTHON_EXECUTABLE = '/mnt/ssd2_4T/ishikawa/miniconda3/envs/fastcomposer/bin/python'

sys.path.append(os.path.abspath(os.getcwd())) 
sys.path.append(GEN_SCRIPTS_DIR) 

env = os.environ.copy()
env['PYTHONPATH'] = f'{GEN_SCRIPTS_DIR}:' + env.get('PYTHONPATH', '')

from data_loader.prompt_loader import DataSaver, DataLoader, load_yaml_config

config = load_yaml_config(yaml_path="./gen_scripts/04_fastcomposer_prompt_config.yaml")

prompt_type_list = ["simple", "action+layout", "action+expression", "action+background", "all"]
# mode_list = ["easy", "medium", "hard"]
mode_list = ["hard"]

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
output_dir_path = config["OPTION"]["output_dir_path"]
test_reference_folder_one_man = config["OPTION"]["test_reference_folder_one_man"]
test_reference_folder_one_woman = config["OPTION"]["test_reference_folder_one_woman"]
test_reference_folder_two = config["OPTION"]["test_reference_folder_two"]

for mode, prompt_type in itertools.product(mode_list, prompt_type_list):

    dir_name = os.path.join(output_dir_path, f"{mode}_{prompt_type}")
    os.makedirs(dir_name, exist_ok = True)
    
    index_list = range(dataloader.get_len_of_data(mode)) if config["index_list"] == None else config["index_list"]
    for idx in index_list:
        data = dataloader.get_idx_info(mode, prompt_type, idx)
        id_ = data["id"]
        p1_sex = data["p1_sex"]
        p2_sex = data["p2_sex"]
        prompt = data["prompt_token"]
        
        if mode=="easy":
            #================
            if p1_sex == "man":
                folder_path = test_reference_folder_one_man
            else:
                folder_path = test_reference_folder_one_woman
            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/fastcomposer/inference.py',
                '--pretrained_model_name_or_path', 'runwayml/stable-diffusion-v1-5',
                '--finetuned_model_path', 'model/fastcomposer',
                '--test_reference_folder', folder_path,
                '--test_caption', prompt,
                '--output_dir', dir_name,
                '--mixed_precision', 'bf16',
                '--image_encoder_type', 'clip',
                '--image_encoder_name_or_path', 'openai/clip-vit-large-patch14',
                '--num_image_tokens', '1',
                '--max_num_objects', '2',
                '--object_resolution', '512',
                '--generate_height', '512',
                '--generate_width', '512',
                '--num_images_per_prompt', '1',
                '--num_rows', '1',
                '--seed', '42',
                '--guidance_scale', '5',
                '--inference_steps', '50',
                '--start_merge_step', '10',
                '--no_object_augmentation',
                '--prompt_id', f'{id_}'
            ], capture_output=True, text=True, env=env)
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
        else:
            folder_path = test_reference_folder_two
            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/fastcomposer/inference.py',
                '--pretrained_model_name_or_path', 'runwayml/stable-diffusion-v1-5',
                '--finetuned_model_path', 'model/fastcomposer',
                '--test_reference_folder', folder_path,
                '--test_caption', prompt,
                '--output_dir', dir_name,
                '--mixed_precision', 'bf16',
                '--image_encoder_type', 'clip',
                '--image_encoder_name_or_path', 'openai/clip-vit-large-patch14',
                '--num_image_tokens', '1',
                '--max_num_objects', '2',
                '--object_resolution', '512',
                '--generate_height', '512',
                '--generate_width', '512',
                '--num_images_per_prompt', '1',
                '--num_rows', '1',
                '--seed', '42',
                '--guidance_scale', '5',
                '--inference_steps', '50',
                '--start_merge_step', '10',
                '--no_object_augmentation',
                '--prompt_id', f'{id_}'
            ], capture_output=True, text=True, env=env)
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")

        datasaver.append_prompt(
            mode=mode, 
            prompt_type=prompt_type, 
            data={
                "id": int(id_),
                "prompt": prompt
            }
        )
    datasaver.save_prompt(os.path.join(output_dir_path, "prompt.json"))


