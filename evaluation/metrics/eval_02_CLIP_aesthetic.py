"""
https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
"""
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
import os
import sys
import argparse
import itertools
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import load_yaml_config, get_dataloader
from evaluation.utils.eval_utils import get_csv_path, save_df_to_csv, image_loader, get_gen_output_path
import pandas as pd


METRIC_NAME = "CLIP_AESTHETIC"


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CLIP_Aesthetic:
    def __init__(self, device: str = "cuda", config: dict = {}):
        self.device = device
        self.model_MLP = MLP(768)
        self.model_MLP.load_state_dict(
            torch.load(config["mlp_model_path"], 
            map_location=self.device)
        )
        self.model_MLP.eval()
        self.model_MLP.to(self.device)
        self.model_ViT = transformers.CLIPModel.from_pretrained(config["clip_model_name"]).to(self.device) # type: ignore
        self.processor = transformers.CLIPProcessor.from_pretrained(config["clip_model_name"])
        self.force_resize = config["force_resize"]

    def __call__(
        self, 
        image_list: Image.Image | list[Image.Image]
    ) -> list[float]:

        if isinstance(image_list, Image.Image):
            image_list = [image_list]

        if self.force_resize:
            image_list = [image.resize((224, 224)) for image in image_list]

        aesthetic_scores = []
        for image in image_list:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device) # type: ignore
            with torch.no_grad():
                image_feature = self.model_ViT.get_image_features(
                    pixel_values=inputs["pixel_values"] # type: ignore
                )
            im_emb_arr = self.normalized(image_feature.unsqueeze(0))
            aesthetic_score = self.model_MLP(im_emb_arr).item()
            aesthetic_scores.append(aesthetic_score)
        return aesthetic_scores
    
    def normalized(
        self,
        a: torch.Tensor, 
        dim: int = -1, 
        order: int = 2
        ) -> torch.Tensor:

        l2 = torch.linalg.norm(a, ord=order, dim=dim, keepdim=True)
        l2 = torch.where(l2 == 0, torch.tensor(1.0, device=a.device, dtype=a.dtype), l2) 
        return a / l2
    


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

    # load config
    config = load_yaml_config(yaml_path=args.yaml_path)
    if args.gen_method != "":
        gen_method_list = [args.gen_method]
    else:
        gen_method_list = config["gen_method_list"]

    dataloader = get_dataloader(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============================
    
    print(f"Evaluating {METRIC_NAME}...")
    model = CLIP_Aesthetic(
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
                p1_sex = prompt_info["p1_sex"]
                p2_sex = prompt_info["p2_sex"]

                generated_img_path = get_gen_output_path(
                    config["gen_output_dir"], 
                    gen_method, 
                    mode, 
                    prompt_type, 
                    id_
                )
                generated_pil = image_loader(generated_img_path)

                # ===============================
                score = model(image_list=generated_pil)
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
