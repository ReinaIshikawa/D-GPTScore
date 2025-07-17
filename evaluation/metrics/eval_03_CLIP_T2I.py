import transformers
import torch
import os
import sys
import argparse
import itertools
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import load_yaml_config, get_dataloader
from evaluation.utils.eval_utils import get_csv_path, save_df_to_csv, image_loader, calc_similarity, get_gen_output_path
import pandas as pd
from PIL import Image

METRIC_NAME = "CLIP_TEXT2IMAGE"

class CLIP_T2I:
    def __init__(
        self,
        device: str = "cuda",
        config: dict = {},
    ):
        self.device = device
        self.force_resize = config["force_resize"]
        model_name_or_path = config["clip_model_name"]
        self.model = transformers.CLIPModel.from_pretrained(model_name_or_path).to(self.device) # type: ignore
        self.processor = transformers.CLIPProcessor.from_pretrained(model_name_or_path)
        

    def __call__(
        self,
        text_list: list[str] | str,
        image_list: list[Image.Image] | Image.Image,
    ) -> list[float]:

        if isinstance(text_list, str):
            text_list = [text_list]
        if isinstance(image_list, Image.Image):
            image_list = [image_list]

        if self.force_resize:
            image_list = [image.resize((224, 224)) for image in image_list]

        similarity_scores = []
        for text, image in zip(text_list, image_list):
            inputs = self.processor(
                text=text, 
                images=image, 
                return_tensors="pt", 
                padding=True) # type: ignore
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            score = calc_similarity(text_embeds, image_embeds).cpu().numpy().tolist()
            similarity_scores.append(score)
        return similarity_scores

"""
what is done in the processor:
        - Resize images to 224
        - Center crop images to 224x224
        - Convert images from grayscale to RGB
        - Normalize pixel values to [0, 1]
        - convert images to tensor
"""

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python 00_eval/eval_CLIP_T2I.py 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', 
        type=str, 
        default="./evaluation/metrics/eval_config.yaml"
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

    # ===============================
    
    print(f"Evaluating {METRIC_NAME}...")
    model = CLIP_T2I(
        device=device, 
        config=config[METRIC_NAME])
    
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
                generated_img_path = get_gen_output_path(
                    config["gen_output_dir"], 
                    gen_method, 
                    mode, 
                    prompt_type, 
                    id_
                )
                generated_pil = image_loader(generated_img_path)

                # ===============================
                score = model(
                    text_list=prompt_token, 
                    image_list=generated_pil
                )
                data = {
                    "mode": mode,
                    "prompt_type": prompt_type,
                    "id_": id_,
                    "score": score[0]
                }
                print(data)
                # ===============================
                save_df_to_csv(output_csv_path, pd.DataFrame([data]))
    print(f"{METRIC_NAME} done.")



