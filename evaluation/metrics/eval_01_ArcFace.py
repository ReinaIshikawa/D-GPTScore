from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os
import sys
import argparse
import itertools
sys.path.append(os.path.abspath(os.getcwd()))
from data_loader.prompt_loader import load_yaml_config, get_dataloader
from PIL import Image
from evaluation.utils.eval_utils import get_csv_path, save_df_to_csv, get_gen_output_path
import pandas as pd

METRIC_NAME = "ArcFace"
#https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
#https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/README.md

def detect_faces(img_path):
    mtcnn = MTCNN(
        margin=40,
        select_largest=False, 
        device='cuda', 
        keep_all=True,  
        post_process=False
    )
    model = InceptionResnetV1(pretrained='vggface2').eval()
    img = Image.open(img_path)
    # boxes, probs = mtcnn.detect(img)
    faces = mtcnn(img)
    embeddings = []
    if faces is not None:
        print(f"{len(faces)} faces were detected")
        for i, face in enumerate(faces):
            embeddings.append(model(face.unsqueeze(0)))
        embeddings = [emb.detach().numpy() for emb in embeddings]
    else:
        print(f"No face was detected")
    return embeddings

def compare_faces(emb1, emb2): 
    """Compare two embeddings using cosine similarity"""
    emb1=emb1.squeeze()
    emb2=emb2.squeeze()
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


if __name__ == "__main__":
    
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

    man_ref_embs = detect_faces(config["man_ref_image"])
    woman_ref_embs = detect_faces(config["woman_ref_image"])
    if man_ref_embs == None or woman_ref_embs == None:
        raise ValueError("No detectable faces in ref image")
    man_ref_emb = man_ref_embs[0]
    woman_ref_emb = woman_ref_embs[0]

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
                p1_sex = prompt_info["p1_sex"]
                p2_sex = prompt_info["p2_sex"]
                generated_img_path = get_gen_output_path(
                    config["gen_output_dir"], 
                    gen_method, 
                    mode, 
                    prompt_type, 
                    id_
                )
                generated_embs = detect_faces(generated_img_path)
                p1_sim_score = None
                p2_sim_score = None

                if len(generated_embs)> 0:
                    if mode == "easy":
                        emb = man_ref_emb if p1_sex == "man" else woman_ref_emb
                        p1_sim_score = compare_faces(generated_embs[0], emb)
                    else:
                        p1_sc_man= compare_faces(generated_embs[0], man_ref_emb)
                        p1_sc_woman= compare_faces(generated_embs[0], woman_ref_emb)
                        p1_sim_score = max(p1_sc_man, p1_sc_woman)

                        if len(generated_embs) > 1:
                            p2_sc_man= compare_faces(generated_embs[1], man_ref_emb)
                            p2_sc_woman= compare_faces(generated_embs[1], woman_ref_emb)
                            p2_sim_score = max(p2_sc_man, p2_sc_woman)

                data = {
                    "mode": mode,
                    "prompt_type": prompt_type,
                    "id_": id_,
                    "p1_sim_score": p1_sim_score,
                    "p2_sim_score":p2_sim_score,
                    "detected_gen": len(generated_embs)
                }
                print(data)

                save_df_to_csv(output_csv_path, pd.DataFrame([data]))