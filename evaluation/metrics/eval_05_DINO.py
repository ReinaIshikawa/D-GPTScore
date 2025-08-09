from transformers import ViTImageProcessor, ViTModel
import torchmetrics
import torch
import numpy as np
import os
import sys
import argparse
import itertools
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import load_yaml_config, get_dataloader
from evaluation.utils.eval_utils import get_csv_path, save_df_to_csv, image_loader, get_gen_output_path
import pandas as pd

METRIC_NAME = "DINO"


class DINOEvaluator:
    def __init__(self, device) -> None:
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

    @torch.inference_mode()
    def get_image_features(self, image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(device=self.device)
        features = self.model(**inputs).last_hidden_state.mean(dim=1)
        return features

    @torch.inference_mode()
    def img_to_img_similarity(self, src_image, generated_image):
        src_features = self.get_image_features(src_image)
        gen_features = self.get_image_features(generated_image)

        return torchmetrics.functional.pairwise_cosine_similarity(src_features, gen_features).mean().item()


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
    args = parser.parse_args()
    config = load_yaml_config(yaml_path=args.yaml_path)
    if args.gen_method != "":
        gen_method_list = [args.gen_method]
    else:
        gen_method_list = config["gen_method_list"]

    dataloader = get_dataloader(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_evaluator = DINOEvaluator(device)

    # ===============================

    man_ref_image = image_loader(config["man_ref_image"])
    woman_ref_image = image_loader(config["woman_ref_image"])

    print(f"Evaluating {METRIC_NAME}...")

    for gen_method in gen_method_list:
        output_csv_path = get_csv_path(
            results_dir=config["results_dir"], 
            metric_name=METRIC_NAME, 
            gen_method=gen_method
        )

        for mode, prompt_type in itertools.product(config["mode_list"], config["prompt_type_list"]):

            index_list = range(dataloader.get_len_of_data(mode)) if config["index_list"] == None else config["index_list"]

            for num, idx in enumerate(index_list):
                print("===============")
                print(f"mode:{mode} / prompt_type:{prompt_type}")

                prompt_info = dataloader.get_idx_info(mode, prompt_type, idx)
                id_ = prompt_info["id_"]
                prompt_token = prompt_info["prompt_token"]
                p1_sex = prompt_info["p1_sex"]
                p2_sex = prompt_info["p2_sex"]
                generated_img_path = get_gen_output_path(
                    config["gen_output_dir"], 
                    gen_method, 
                    mode, 
                    prompt_type, 
                    id_
                )
                generated_image = image_loader(generated_img_path)

                # ===============================

                if mode == "easy":
                    ref_1 = man_ref_image if p1_sex == "man" else woman_ref_image
                    score = dino_evaluator.img_to_img_similarity(
                        ref_1, 
                        generated_image)
                else:
        
                    man_score = dino_evaluator.img_to_img_similarity(
                        man_ref_image, 
                        generated_image)
                    woman_score = dino_evaluator.img_to_img_similarity(
                        woman_ref_image, 
                        generated_image)
                    score = np.mean([man_score, woman_score])

                data = {
                    "mode": mode,
                    "prompt_type": prompt_type,
                    "id_": id_,
                    "score": score
                }
                print(data)
                # ===============================
                save_df_to_csv(output_csv_path, pd.DataFrame([data]))
    print(f"{METRIC_NAME} done.")

    