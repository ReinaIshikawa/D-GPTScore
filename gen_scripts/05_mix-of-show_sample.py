import subprocess
import sys
import os
import itertools
import torch
from diffusers import DPMSolverMultistepScheduler

GEN_SCRIPTS_DIR = os.path.expanduser("~/data/Mix-of-Show")
PYTHON_EXECUTABLE = '/mnt/ssd2_4T/ishikawa/miniconda3/envs/mix-of-show/bin/python'


sys.path.append(os.path.abspath(os.getcwd())) 
sys.path.append(GEN_SCRIPTS_DIR) 

env = os.environ.copy()
env['PYTHONPATH'] = f'{GEN_SCRIPTS_DIR}:' + env.get('PYTHONPATH', '')

from data_loader.prompt_loader import DataSaver, DataLoader, load_yaml_config
from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline
from mixofshow.utils.convert_edlora_to_diffusers import convert_edlora

config = load_yaml_config(yaml_path="./gen_scripts/05_mix-of-show_prompt_config.yaml")

prompt_type_list = ["simple", "action+layout", "action+expression", "action+background", "all"]
mode_list = ["easy", "medium", "hard"]
# prompt_type_list = ["action+expression", "all"]
# mode_list = ["hard"]

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

option = config["OPTION"]
neg_prompt = option["neg_prompt"]
output_dir_path = option["output_dir_path"]
single_pretrained_model_path = option["single_pretrained_model_path"]
single_lora_model_path_man = option["single_lora_model_path_man"]
single_lora_model_path_woman = option["single_lora_model_path_woman"]
multi_pretrained_model_path = option["multi_pretrained_model_path"]
region_pt1 = option["region_pt1"]
region_pt2 = option["region_pt2"]

pipeclass = EDLoRAPipeline
single_pipe_man = pipeclass.from_pretrained(
    single_pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(
        single_pretrained_model_path, 
        subfolder='scheduler'
    ), torch_dtype=torch.float16
).to('cuda')
single_pipe_man, new_concept_cfg = convert_edlora(
    single_pipe_man, 
    torch.load(single_lora_model_path_man), 
    enable_edlora=True, 
    alpha=1.0
)
single_pipe_man.set_new_concept_cfg(new_concept_cfg)

single_pipe_woman = pipeclass.from_pretrained(
    single_pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(
        single_pretrained_model_path, 
        subfolder='scheduler'
    ), torch_dtype=torch.float16
).to('cuda')
single_pipe_woman, new_concept_cfg = convert_edlora(
    single_pipe_woman, 
    torch.load(single_lora_model_path_woman), 
    enable_edlora=True, 
    alpha=1.0
)
single_pipe_woman.set_new_concept_cfg(new_concept_cfg)



for mode, prompt_type in itertools.product(mode_list, prompt_type_list):

    dir_name = os.path.join(output_dir_path, f"{mode}_{prompt_type}")
    os.makedirs(dir_name, exist_ok = True)
    
    index_list = range(dataloader.get_len_of_data(mode)) if config["index_list"] == None else config["index_list"]

    for idx in index_list:
        data = dataloader.get_idx_info(mode, prompt_type, idx)
        id_ = data["id"]
        prompt_token = data["prompt_token"]
        prompt_class = data["prompt_class"]
        pt1 = data["pt1"]
        pt2 = data["pt2"]
        p1_sex = data["p1_sex"]
        p2_sex = data["p2_sex"]

        rewrite = ""
        
        if mode=="easy":
            prompt = prompt_token
            #================
            if p1_sex == "man":
                pipe = single_pipe_man
            else:
                pipe = single_pipe_woman
            #================
            pipe.unet.eval()
            pipe.text_encoder.eval()
            image=pipe(
                prompt, 
                negative_prompt=neg_prompt, 
                height=512, 
                width=512, 
                num_inference_steps=50, 
                guidance_scale=7.5
            ).images[0]
            image.save(os.path.join(dir_name, f'{id_:03}.png'))
        else:
            prompt = prompt_class
            rewrite = f"[{pt1}]-*-[{neg_prompt}]-*-{region_pt1}|[{pt2}]-*-[{neg_prompt}]-*-{region_pt2}"
            result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/regionally_controlable_sampling.py',
                '--pretrained_model', multi_pretrained_model_path,
                '--sketch_adaptor_weight', '0',
                '--sketch_condition', '',
                '--keypose_adaptor_weight', '0',
                '--keypose_condition', '',
                '--prompt', prompt,
                '--negative_prompt', neg_prompt,
                '--prompt_rewrite', rewrite,
                '--save_dir', dir_name,
                '--prompt_id', f'{id_}',
                '--suffix', 'baseline',
                '--seed', '19'
            ], capture_output=True, text=True, env=env)
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


