# D-GPTScore (ICCVW2025)
Official implementation of "Human Preference-Aligned Concept Customization Benchmark via Decomposed Evaluation."


# Getting Started

### Installation
```bash
git clone git@github.com:ReinaIshikawa/D-GPTScore.git
conda env create -f environment.yml
conda activate d-gpt
```

To use the OpenAI API, run the following command in your terminal:
```bash
echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
```

### Dataset Structure
- 20 images of each person in `CC-AlignBench/man_1` and `CC-AlignBench/woman_1`
- Text prompts in `CC-AlignBench/cc-alignbench-data.csv`

### Data Loading Example
You can use `data_loader/prompt_loader.py`.

Example:
```bash
python data_loader/prompt_loader.py \
 --csv_path ./CC-AlignBench/data.csv \
 --man_token "A*" \
 --woman_token "B*" \
 --debug \
 --index_list 46 \
 --prompt_type action+expression \
 --mode easy
```

For detailed information about the options, please check the script.

## Evaluating D-GPTScore on a Generated Image

```bash
python evaluation/metrics/eval_end-to-end.py \
 --generated_img_path <path/to/the/generated/image> \
 --concept_path <path/to/the/one/or/two/concept/images> \
 --prompt <prompt/used/to/generate/the/image> \
```

Example:

```bash
python evaluation/metrics/eval_end-to-end.py \
 --generated_img_path ./results/eval_end-to-end/generated_img.png \
 --concept_path ./CC-AlignBench/man_1/image/0.jpeg ./CC-AlignBench/woman_1/image/0.jpeg \
 --prompt "a man and a woman, shaking hands" \
```

## Evaluating D-GPTScore of Your Model on CC-AlignBench

1. **Save images generated with CC-AlignBench**

Prepare `sample.py` in `./gen_scripts` to generate images.
You can refer to the example scripts in the same directory.

`sample.py`:
```python
import subprocess
import os
import itertools
import sys

# ==================================================
# Please set the following parameters
# We added some modifications to the generation scripts to make them work with our code, especially for saving the generated images

GEN_SCRIPTS_DIR = os.path.expanduser("~/path/to/your/model/dir")
PYTHON_EXECUTABLE = 'path/to/the/executable/python'
output_dir_path = "./gen_output/<method_name>/output"
man_token = "<token to replace '<|man_1|>' in the prompt>"
woman_token = "<token to replace '<|woman_1|>' in the prompt.>"
# ==================================================

sys.path.append(os.path.abspath(os.getcwd())) 
sys.path.append(GEN_SCRIPTS_DIR) 

from data_loader.prompt_loader import DataSaver, CCAlignBenchLoader, load_yaml_config

env = os.environ.copy()
env['PYTHONPATH'] = f'{GEN_SCRIPTS_DIR}:' + env.get('PYTHONPATH', '')

prompt_type_list = ["simple", "action+layout", "action+expression", "action+background", "all"]
mode_list = ["easy", "medium", "hard"]

csv_path = "./CC-AlignBench/cc-alignbench-data.csv"
dataloader = CCAlignBenchLoader(
    csv_path = csv_path,
    man_token = man_token, 
    woman_token = woman_token)

for mode, prompt_type in itertools.product(mode_list, prompt_type_list):

    dir_name = os.path.join(output_dir_path, f"{mode}_{prompt_type}")
    os.makedirs(dir_name, exist_ok = True)

    index_list = range(dataloader.get_len_of_data(mode)) 
    for idx in index_list:
        data = dataloader.get_idx_info(mode, prompt_type, idx)
        id_ = data["id"]
        prompt = data["prompt_token"]


        # =============
        # Generate image and save it as f"{idx}.png" under "dir_name" defined above.
        # =============
        result=subprocess.run([
                PYTHON_EXECUTABLE, f'{GEN_SCRIPTS_DIR}/src/diffusers_sample.py', 
                '--prompt', prompt, 
                '--outdir', dir_name,
                '--prompt_id', f'{id_}'
                ...
            ], capture_output=True, text=True, env=env)
```

2. **Execute sampling:**

```bash
$ python gen_scripts/sample.py
```

3. **Get D-GPTScore**

```bash
$ python evaluation/analysis/benchmark_raderchart.py --gen-method 01_CustomDiffusion
```
The result will be saved as `./evaluation/results/output/wo_r_GPT_ours_<method_name>_output.csv`

If you want to evaluate for each difficulty mode, you can use the `--mode` and `--type` options.


## Paper Results
Currently, the following results are available:

- CustomDiffusion
- OMG(LoRA)
- OMG(InstantID)
- FastComposer
- Mix-of-Show
- DreamBooth

The generation results should be saved under `gen_output/(method name)/output`

```bash
python evaluation/analysis/corr_02_ours_ave.py
```
The result will be saved in `evaluation/results/GPT_ours_ave/`



