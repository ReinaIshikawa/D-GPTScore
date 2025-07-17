from PIL import Image
import base64
from io import BytesIO
import os
import itertools
import pandas as pd
from evaluation.utils.eval_utils import save_df_to_csv, calc_mean_and_std
from evaluation.utils.prompts import EvaluationData, Reason_text, Result_w_rationale, Result_wo_rationale, HEADER_w_tex_wo_ref, HEADER_wo_tex_w_ref, HEADER_wo_tex_wo_ref, HEADER_wo_tex_wo_ref_crop, VanillaGPTMessage

def encode_image(image_path: str, crop_flag: str | None = None) -> str:
    with Image.open(image_path) as image:
        if crop_flag== "center":
            image = image.crop((256, 256, 768, 768))
        if crop_flag== "left":
            image = image.crop((0, 256, 512, 768))
        if crop_flag== "right":
            image = image.crop((512, 256, 1024, 768))
        image = image.resize((512, 512))
        buffered = BytesIO()
        image.save(buffered, format="png")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

class GPTMessageGenerator():
    def __init__(self):
        self.content = []

    def add_text_to_content(self, new_text: str):
        new_content = {
            "type": "text", 
            "text": new_text
        }
        self.content.append(new_content)

    def add_base64_image(self, image_path: str, crop_flag: str | None = None):
        ext = os.path.splitext(image_path)[1][1:]
        image_base64 = encode_image(image_path, crop_flag)
        new_content = (
        {
            "type": "image_url",
            "image_url":{"url": f"data:image/{ext};base64,{image_base64}"},
        }
        )
        self.content.append(new_content)
    
    def get_messages(self) -> list[dict]:
        messages = [
            {
                "role": "user",
                "content": self.content
            },
        ]
        return messages

    
def get_text_prompt_part(
        ea_id: int, 
        textprompt: str, 
        reason_flag: bool = False
    ) -> str:
    """
    idx: int[1-18]
    """
    data = EvaluationData[f"{ea_id}"]
    eval_aspect = data["eval_aspect"]
    replaced_reason_tex = Reason_text if reason_flag else ""

    if data["text_req"]:
        eval_aspect = eval_aspect.format(textprompt)
    if data["text_req"] and not data["ref_image_req"]:
        header = HEADER_w_tex_wo_ref.format(replaced_reason_tex)
    elif not data["text_req"] and data["ref_image_req"]:
        header = HEADER_wo_tex_w_ref.format(replaced_reason_tex)
    elif not data["text_req"] and not data["ref_image_req"]:
        if ea_id in [6,14,15,16,17]:
            header = HEADER_wo_tex_wo_ref_crop.format(replaced_reason_tex)
        else:
            header = HEADER_wo_tex_wo_ref.format(replaced_reason_tex)
    gpt_prompt = header+eval_aspect
    return gpt_prompt

def get_messages_part(
    prompt_token: str, 
    ref_image_path1: str,
    ref_image_path2: str | None,
    generated_img_path: str,
    ea_id: int,
    reason_flag: bool
):

    """
    prompt_token: the prompt used to generate the image
    ref_image_path1: the first reference image
    ref_image_path2: the second reference image
    generated_img_path: the generated image
    ea_id: the id of the evaluation aspect
    reason_flag: whether to include the rationale in the response
    """


    message_generator = GPTMessageGenerator()

    # add the texts including text prompt if required
    message_generator.add_text_to_content(
        get_text_prompt_part(ea_id, prompt_token, reason_flag)
    )

    # add the generated image
    image_file_paths=[generated_img_path]

    # add the reference images if required
    if EvaluationData[f"{ea_id}"]["ref_image_req"]:
        image_file_paths.append(ref_image_path1)
        if ref_image_path2 is not None:
            image_file_paths.append(ref_image_path2)
        
    for i, f in enumerate(image_file_paths):
        if ea_id in [6, 14,15,16,17] and i==0:
            message_generator.add_base64_image(f, "left")
            message_generator.add_base64_image(f, "right")
        message_generator.add_base64_image(f)

    # add "resut:" text
    result_tex = Result_w_rationale if reason_flag else Result_wo_rationale
    message_generator.add_text_to_content(result_tex)
    messages = message_generator.get_messages()
    return messages


def get_messages_vanilla(
    prompt_token: str, 
    ref_image_path1: str,
    ref_image_path2: str | None,
    generated_img_path: str,
    reason_flag: bool
    ):

    message_generator = GPTMessageGenerator()

    replaced_reason_tex = Reason_text if reason_flag else ""
    vanillaMessage = VanillaGPTMessage.format(replaced_reason_tex, prompt_token)
    message_generator.add_text_to_content(vanillaMessage)

    # add the generated image
    image_file_paths=[generated_img_path]
    # add the reference image1
    image_file_paths.append(ref_image_path1)
    # add the reference image2 if required
    if ref_image_path2 is not None:
        image_file_paths.append(ref_image_path2)

    # add the images to the message
    for f in image_file_paths:
        message_generator.add_base64_image(f)

    # add "resut:" text
    result_tex = Result_w_rationale if reason_flag else Result_wo_rationale
    message_generator.add_text_to_content(result_tex)
    messages = message_generator.get_messages()

    return messages

def calc_and_save_mean_and_std(
        output_csv_path,
        analysis_mean_csv_path,
        analysis_std_csv_path,
        mode_list,
        prompt_type_list, 
        target_cols):
    df = pd.read_csv(output_csv_path)

    for mode, prompt_type in itertools.product(mode_list, prompt_type_list):
        print(f"- mode:{mode} / prompt_type:{prompt_type}")

        mean_df, std_df = calc_mean_and_std(
            df, mode, prompt_type, target_cols
        )
        save_df_to_csv(analysis_mean_csv_path, mean_df)
        save_df_to_csv(analysis_std_csv_path, std_df)
