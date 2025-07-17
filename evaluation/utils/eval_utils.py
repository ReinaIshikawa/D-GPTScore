import torch
from PIL import Image
import os
import pandas as pd


def calc_similarity(
        embeds_out: torch.Tensor,
        embeds_gt: torch.Tensor,
    ) -> torch.Tensor:

    embeds_out = embeds_out / embeds_out.norm(dim=-1, keepdim=True)
    embeds_gt = embeds_gt / embeds_gt.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(embeds_out, embeds_gt.T)
    return similarity.squeeze()

def save_df_to_csv(
        csv_path: str, 
        df: pd.DataFrame
    ) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        df.to_csv(file, header=not file_exists, index=False)

def calc_mean_and_std(
        df: pd.DataFrame, 
        mode: str, 
        prompt_type: str, 
        target_cols: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    if "mode" not in df or "prompt_type" not in df:
        raise KeyError(f" mode or prompt_type column is missing in the data dictionary.")
    df_filtered = df[(df["mode"] == mode) & (df["prompt_type"] == prompt_type)].copy()

    mean_values = df_filtered[target_cols].mean()
    mean_df = pd.DataFrame([mean_values])
    mean_df["mode"] = mode
    mean_df["prompt_type"] = prompt_type
    mean_df = mean_df[["mode", "prompt_type"] + list(mean_values.keys())] # type: ignore

    std_values = df_filtered[target_cols].std()
    std_df = pd.DataFrame([std_values])
    std_df["mode"] = mode
    std_df["prompt_type"] = prompt_type
    std_df = std_df[["mode", "prompt_type"] + list(std_values.keys())] # type: ignore
    return mean_df, std_df # type: ignore

def get_csv_path(
        results_dir: str, 
        metric_name: str, 
        gen_method: str, 
        header: str = ""
    ) -> str:
    output_dir = os.path.join(
        results_dir,
        "output"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(
        output_dir, 
        f"{header}{metric_name}_{gen_method}_output.csv"
    )

    return output_csv_path

def get_gen_output_path(
        gen_output_dir: str, 
        gen_method: str, 
        mode: str, 
        prompt_type: str, 
        id_: int
    ) -> str:
    output_path = os.path.join(
        gen_output_dir, 
        gen_method, 
        "output", 
        f"{mode}_{prompt_type}", 
        f"{id_:03}.png"
    )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output path not found: {output_path}")
    return output_path

    
def image_loader(image_path: str) -> Image.Image:
    with open(image_path, "rb") as f:
        pil_image = Image.open(f).convert("RGB")
        if pil_image is None:
            raise ValueError(f"Invalid image: {image_path}")
    return pil_image