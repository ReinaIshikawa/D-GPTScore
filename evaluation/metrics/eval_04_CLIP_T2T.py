import transformers
import torch
from PIL import Image
import os
import sys
import argparse
import itertools
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import load_yaml_config, get_dataloader
from evaluation.utils.eval_utils import get_csv_path, save_df_to_csv, image_loader, calc_similarity, get_gen_output_path
import pandas as pd

METRIC_NAME = "CLIP_TEXT2TEXT"


class BLIP_CAPTION:
    def __init__(self, device: str = "cuda", config: dict = {}):
        self.device = device
        self.force_resize = config["force_resize"]

        if config["blip_model_name"] == "base":
            model_name_or_path = "Salesforce/blip-image-captioning-base"
        elif config["blip_model_name"] == "large":
            model_name_or_path = "Salesforce/blip-image-captioning-large"
        else:
            raise ValueError(f"Invalid model name: {config['blip_model_name']}")

        self.model = transformers.BlipForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device) # type: ignore
        self.processor = transformers.BlipProcessor.from_pretrained(model_name_or_path)

    def __call__(
        self,
        image_list: list[Image.Image] | Image.Image,
    ) -> list[str]:

        if isinstance(image_list, Image.Image):
            image_list = [image_list]

        if self.force_resize:
            image_list = [image.resize((224, 224)) for image in image_list]

        inputs = self.processor(images=image_list, return_tensors="pt").to(self.device) # type: ignore
        with torch.no_grad():
            output = self.model.generate(pixel_values=inputs["pixel_values"]) # type: ignore
        captions = [self.processor.decode(out, skip_special_tokens=True) for out in output] # type: ignore

        return captions

class CLIP_T2T:
    def __init__(
        self,
        device: str = "cuda",
        config: dict = {},
    ):
        self.device = device
        self.force_resize = config["force_resize"]

        self.model = transformers.CLIPModel.from_pretrained(config["clip_model_name"]).to(self.device) # type: ignore
        self.processor = transformers.CLIPProcessor.from_pretrained(config["clip_model_name"])
        

    def __call__(
        self,
        text_gt: list[str] | str,
        text_out: list[str] | str,
    ) -> list[float]:

        if isinstance(text_gt, str):
            text_gt = [text_gt]
        if isinstance(text_out, str):
            text_out = [text_out]

        similarity_scores = []
        for gt, out in zip(text_gt, text_out):
            gt_input = self.processor(text=gt, return_tensors="pt", padding=True).to(self.device) # type: ignore
            out_input = self.processor(text=out, return_tensors="pt", padding=True).to(self.device) # type: ignore
            with torch.no_grad():
                gt_embeds = self.model.get_text_features(**gt_input) # type: ignore
                out_embeds = self.model.get_text_features(**out_input) # type: ignore

            score = calc_similarity(out_embeds, gt_embeds).cpu().numpy().tolist()
            similarity_scores.append(score)
        return similarity_scores

class CLIP_T2T_WITH_BLIP:
    def __init__(
        self,
        device: str = "cuda",
        config: dict = {},
    ):
        self.device = device
        self.config = config
        self.force_resize = config["force_resize"]
        
        self.blip_model = BLIP_CAPTION(device=device, config=config)
        self.clip_model = CLIP_T2T(device=device, config=config)

    def __call__(
        self,
        images: list[Image.Image] | Image.Image,
        texts: list[str] | str,
    ) -> tuple[list[float], list[str]]:

        blip_captions = self.blip_model(images)
        clip_scores = self.clip_model(texts, blip_captions)
        return clip_scores, blip_captions


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python 00_eval/eval_CLIP_T2T.py 
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

    # ===============================
    
    print(f"Evaluating {METRIC_NAME}...")
    model = CLIP_T2T_WITH_BLIP(
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
                id_ = prompt_info["id"]
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
                score, blip_caption = model(
                    texts=prompt_token, 
                    images=generated_pil
                )
                data = {
                    "mode": mode,
                    "prompt_type": prompt_type,
                    "id_": id_,
                    "score": score[0],
                    "blip_caption":blip_caption[0]
                }
                print(data)
                # ===============================
                save_df_to_csv(output_csv_path, pd.DataFrame([data]))
    print(f"{METRIC_NAME} done.")
