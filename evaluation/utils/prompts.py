
VanillaGPTMessage = """
Task:
I will provide a text prompt, followed by a generated image and one or two reference images. Please evaluate the generated image and assign a score on a scale from 1 to 10{0}. Pay attention to whether the characteristics of the individuals in the reference images (including clothing, etc.) are preserved and whether the generated image follows the text prompt.

The text prompt:
"{1}"
"""

EvaluationData ={
    "1":{
        "id":"EA1",
        "name":"Subject Type",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""

Evaluation aspect:
Do the generated objects and people match the specified types (e.g., 'a man' should not be misrepresented as 'a woman')?  Focus ONLY on the subject type accuracy.

Scoring example:
If the genders are swapped, subtract 4 points.

The text prompt:
"{0}"
"""
    },
    "2":{
        "id":"EA2",
        "name":"Quantity",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are the correct number of objects and persons generated according to the prompt? Focus ONLY on quantity accuracy. 

Scoring example:
If the prompt specifies two men but three are generated, subtract 4 points.

The text prompt:
"{0}"
"""
    },
    "3":{
        "id":"EA3",
        "name":"Sbject & Camera Positioning",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are objects and people positioned correctly and arranged logically within the scene, preserving appropriate spatial relationships, depth, and occlusion according to the specified layout? Focus ONLY on the subject and camera positioning. if there is no relevant part in the text prompt, ignore the prompt.

Scoring example:
If a 'long shot' is required but a close-up is generated, subtract 3 points.
The text prompt:
"{0}"
"""
    },
    "4":{
        "id":"EA4",
        "name":"Size & Scale",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are the absolute and relative sizes of objects and people appropriate for the scene? Focus ONLY on the size and scale. 

Scoring example:
If the man in the image appears too small relative to the surrounding objects, subtract 4 points.

The text prompt:
"{0}"
"""
    },
    "5":{
        "id":"EA5",
        "name":"Color",
        "text_req":False,
        "ref_image_req":True,
        "eval_aspect":"""
Evaluation aspect:
Are the colors applied appropriately according to the reference images? Focus ONLY on the color accuracy.

Scoring example:
If the skin tone or hair color is different from the reference image, subtract 3 points.
"""
    },
    "6":{
        "id":"EA6",
        "name":"Subject Completeness",
        "text_req":False,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Is the object or person fully generated with no missing or extra parts? Focus ONLY on the subject completeness.
*Pay special attention to where the two individuals are in contact*

Scoring example:
If the hands touching the other person are semi-transparent or unclear, subtract 3 points.
"""
    },
    "7":{
        "id":"EA7",
        "name":"Proportions & Body Consistency",
        "text_req":False,
        "ref_image_req":True,
        "eval_aspect":"""
Evaluation aspect:
Are body proportions and limb positioning natural and consistent with the given text prompt or reference image? Focus ONLY on the proportions and body consistency.

Scoring example:
If the limb or arm is unnatural, subtract 4 points; if the body proportions are off, subtract 3 points.
"""
    },
    "8":{
        "id":"EA8",
        "name":"Actions & Expressions",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are specified actions, poses, gaze direction, and facial expressions correctly depicted, reflecting the intended motion and emotion from the text prompt? Focus ONLY on the actions and expressions.

Scoring example:
If the man is instructed to laugh but isn't, subtract 4 points.

The text prompt:
"{0}"
"""
    },
    "9":{
        "id":"EA9",
        "name":"Facial Similarity & Features",
        "text_req":False,
        "ref_image_req":True,
        "eval_aspect":"""
Evaluation aspect:
Does the generated face resemble the reference image, preserving key characteristics like shape, expression, and symmetry? Focus ONLY on the facial similarity and features.

Scoring example:
If the face differs from the reference but keeps key features like hairstyle, subtract 3 points.
"""
    },
    "10":{
        "id":"EA10",
        "name":"Clothing & Attributes",
        "text_req":False,
        "ref_image_req":True,
        "eval_aspect":"""
Evaluation aspect:
Are clothing, accessories, and key features consistent with the reference images? Focus ONLY on the clothing and attributes.

Scoring example:
If the person is missing accessories, subtract 1 point; if the clothing differs completely from the reference, subtract 2 points.
"""
    },
    "11":{
        "id":"EA11",
        "name":"Surroundings",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Is the surrounding environment accurately depicted according to the provided text prompt? Focus ONLY on the surroundings.

Scoring example:
If a cafe is specified but a photo of a park is generated, assign 1 point; if there is no relevant part in the text prompt, ignore the prompt.

The text prompt:
"{0}"
"""
    },
    "12":{
        "id":"EA12",
        "name":"Human & Animal Interactions",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are persons and animals interacting naturally with objects and each other as specified in the text prompt? Focus ONLY on the human and animal interactions.

Scoring example:
If the prompt specifies hugging but the image shows handshaking, subtract 4 points.

The text prompt:
"{0}"
"""
    },
    "13":{
        "id":"EA13",
        "name":"Object Interactions",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are objects interacting logically within the scene as specified in the text prompt? Focus ONLY on the object interactions.

Scoring example:
If the book in the prompt sinks into the table, subtract 4 points.

The text prompt:
"{0}"
"""
    },
    "14":{
        "id":"EA14",
        "name":"Subject Deformation",
        "text_req":False,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are the people in the image (especially the faces and where the two individuals are in contact) rendered without deformations? Focus ONLY on the subject's deformation. 

Scoring example:
If the person's face has any deformation or unrecognizable, subtract 4 points.
"""
    },
    "15":{
        "id":"EA15",
        "name":"Surroundings Deformation",
        "text_req":False,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are the surroundings in the image rendered naturally, without deformations such as crooked lines or unnatural parts? Focus ONLY on the surroundings' deformation.

Scoring example:
If the surroundings have deformation, subtract 4 points.
"""
    },
    "16":{
        "id":"EA16",
        "name":"Local Artifacts",
        "text_req":False,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are the image rendered without unwanted noise, strange patterns, or incomplete renderings? Focus ONLY on the local artifacts.

Scoring example:
If there is an unwanted watermark on the generated image, subtract 3 points. 
"""
    },
    "17":{
        "id":"EA17",
        "name":"Detail & Sharpness",
        "text_req":False,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Are facial features, hands, and intricate details well-defined? Focus ONLY on the detail and sharpness.

Scoring example:
If the entire image lacks detail, subtract 4 points; if a person is missing detail in any part (e.g., hands, legs, arms, face), subtract 2 points.
"""
    },
    "18":{
        "id":"EA18",
        "name":"Style Consistency",
        "text_req":True,
        "ref_image_req":False,
        "eval_aspect":"""
Evaluation aspect:
Does the generated image adhere to the artistic or visual style specified in the text prompt? Focus ONLY on the style consistency.

Scoring example:
If the prompt requires a realistic image but the style is anime-like, subtract 4 points.

The text prompt:
"{0}"
"""
    }
}

HEADER_w_tex_wo_ref="""Task:
I will provide a text prompt, followed by a generated image. Please rate how well the generated image meets the following evaluation aspect{0}, then give a score from 1 to 5.
DO NOT check whether the generated image matches the entire text prompt. Instead, rate it solely based on the following evaluation aspect.
"""


HEADER_wo_tex_w_ref="""Task:
I will provide a generated image, followed by one or two reference images. Please observe carefully how well the generated image meets the following evaluation aspect{0}, then give a score from 1 to 5.
"""

HEADER_wo_tex_wo_ref="""Task:
I will provide a generated image. Please observe carefully how well the generated image meets the following evaluation aspect{0}, then give a score from 1 to 5.
"""

HEADER_wo_tex_wo_ref_crop="""Task:
I will present a set of images cropped at different locations of the same generated image. Please observe carefully how well the generated image meets the following evaluation aspect {0}, then give a score from 1 to 5 based on the worst image.
"""

Result_w_rationale="""
Rationale and score:

"""

Result_wo_rationale="""
Score:

"""

Reason_text = ", along with a rationale within 50 words"