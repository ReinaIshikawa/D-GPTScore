import json
import yaml
import pandas as pd
import itertools
import os

class DataSaver():
    def __init__(self, prompt_type_list, mode_list, config):
        empty_data = {f"{mode}_{prompt_type}": [] for mode,prompt_type  in itertools.product(mode_list, prompt_type_list)}
        self.to_save = {
            "config": config,
            "data":empty_data
        }

    def append_prompt(self, mode, prompt_type, data):
        key = f"{mode}_{prompt_type}"
        self.to_save["data"][key].append(data)

    def save_prompt(self, save_path):
        with open(save_path, mode="w") as f:
            json.dump(self.to_save, f, indent=4)
        print(f"prompts saved to {save_path}")

def get_dataloader(config):
    csv_path = os.path.join(config["dir"],config["csv_file"])

    dataloader = CCAlignBenchLoader(
        csv_path = csv_path,
        man_token = config["man_token"], 
        woman_token = config["woman_token"]
    )
    return dataloader

class CCAlignBenchLoader():
    def __init__(
        self, 
        csv_path = "./CC-AlignBench/cc-alignbench-data.csv",
        man_token = "a man", 
        woman_token = "a woman"
        ):

        self.all_df = pd.read_csv(
            csv_path,
            sep=';', 
            encoding='utf-8'
        ).fillna("")

        self.man_token = man_token
        self.woman_token = woman_token

    def get_len_of_data(self, mode, prompt_type):
        return len(self.all_df[(self.all_df['mode'] == mode) & 
                               (self.all_df['prompt_type'] == prompt_type)])
    
    def get_idx_info(self, mode, prompt_type, idx):
        filtered_df = self.all_df[(self.all_df['mode'] == mode) & 
                                  (self.all_df['prompt_type'] == prompt_type) & 
                                  (self.all_df['id'] == idx)]
        row_df = filtered_df.iloc[0].to_dict()
        id = row_df['id']
        prompt = row_df['prompt']
        p1_sex, p2_sex = row_df["p1_sex"], row_df["p2_sex"]
        p1_prompt, p2_prompt = row_df["p1_prompt"], row_df["p2_prompt"]

        p1_replace = self.man_token if p1_sex == "man" else self.woman_token
        p2_replace = self.man_token if p2_sex == "man" else self.woman_token

        prompt_token = prompt.replace(f"<|{p1_sex}_1|>", p1_replace).replace(f"<|{p2_sex}_1|>", p2_replace)
        prompt_class = prompt.replace(f"<|{p1_sex}_1|>", f"a {p1_sex}").replace(f"<|{p2_sex}_1|>", f"a {p2_sex}")
        p1_prompt = p1_prompt.replace(f"<|{p1_sex}_1|>", f"the {p1_sex}")
        if p2_sex == "" or p2_sex == None:
            p2_prompt = "" 
        else:
            p2_prompt = p2_prompt.replace(f"<|{p2_sex}_1|>", f"the {p2_sex}")

        data = {
            "id": id,
            "prompt_token": prompt_token,
            "prompt_class": prompt_class,
            "p1_prompt": p1_prompt,
            "p2_prompt": p2_prompt,
            "p1_sex": p1_sex,
            "p2_sex": p2_sex,
        }
        return data
    
def load_yaml_config(yaml_path: str = "config.yaml") -> dict:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f'{yaml_path} not found')
        return {}
    except yaml.YAMLError as e:
        print(f'error: {e}')
        return {}

if __name__ == "__main__":

    """
    example:
    $ python concept_customization_LLMEval/prompt_loader.py --man_token "a man" --woman_token "a woman" --index_list 0 9 11 23 34 --debug
    $ python concept_customization_LLMEval/prompt_loader.py --man_token "A*" --woman_token "B*" --debug --index_list 46 --prompt_type action+expression --mode easy
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', type=str, default="", help="path to the yaml file")
    parser.add_argument(
        '--csv_path', type=str, default="", help="path to the csv file")
    parser.add_argument(
        '--man_token', type=str, default="Bob")
    parser.add_argument(
        '--woman_token', type=str, default="Lisa")
    parser.add_argument(
        '--mode', type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument(
        '--prompt_type', type=str, default="all", choices=["all", "action+layout", "action+expression", "action+background", "simple"])
    parser.add_argument(
        '--index_list', type=int, nargs='+')
    args = parser.parse_args()

    if args.yaml_path != "":
        print("yaml_path:", args.yaml_path)
        config = load_yaml_config(yaml_path=args.yaml_path)
        dataloader = get_dataloader(config)

    elif args.csv_path != "":
        print("csv_path:", args.csv_path)
        dataloader = CCAlignBenchLoader(
            csv_path = args.csv_path,
            man_token = args.man_token, 
            woman_token = args.woman_token
        )
    else:
        raise ValueError("csv_path or yaml_path must be provided")


    index_list = range(dataloader.get_len_of_data(args.mode, args.prompt_type)) if args.index_list == None else args.index_list
    print(dataloader.get_len_of_data(args.mode, args.prompt_type))
    for idx in index_list:
        print("mode:", args.mode)
        print("prompt_type:", args.prompt_type)
        data = dataloader.get_idx_info(args.mode, args.prompt_type, idx)
        for key, value in data.items():
            print(f"{key}: {value}")
        print("-"*10)


