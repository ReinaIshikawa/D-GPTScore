import json
import os
import yaml
import pandas as pd
import itertools

def get_dataloader(config):
    csv_path = os.path.join(config["dir"],config["csv_file"])
    bg_path = os.path.join(config["dir"],config["bg_file"])

    dataloader = DataLoader(
        csv_path = csv_path,
        bg_path = bg_path,
        surrounings_type = config["surrounings_type"], 
        man_token = config["man_token"], 
        woman_token = config["woman_token"], 
        debug = config["debug"]
    )
    return dataloader

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

class DataLoader():
    def __init__(
        self, 
        csv_path = "./CC-AlignBench/_data.csv", 
        bg_path = "./CC-AlignBench/_background.json", 
        surrounings_type = "simple", 
        man_token = "a man", 
        woman_token = "a woman",
        debug = False
        ):

        self.all_df = pd.read_csv(
            csv_path,
            sep=';', 
            encoding='utf-8'
        ).fillna("")

        with open(bg_path, 'r', encoding='utf-8') as bf:
            self.bg_dict=json.load(bf)

        self.man_token = man_token
        self.woman_token = woman_token
        self.debug = debug
        self.surrounings_type = surrounings_type
    

    def replace_prompt(self, default_prompt, concept_1="<new1>", concept_2="<new2>"):
        return default_prompt.format(concept_1, concept_2) 
    
    def get_len_of_data(self, mode):
        mode_df = self.all_df.query(f'mode==@mode')
        rows, _ = mode_df.shape
        if self.debug:
            print("length of data: ", rows)
        return rows
    
    def get_row_data(self, mode, idx):
        mode_df = self.all_df.query(f'mode==@mode')
        row_df = mode_df.iloc[idx].to_dict()
        p1_sex, p2_sex = row_df["person1_sex"], row_df["person2_sex"]
        row_df["p1_token"] = self.man_token if p1_sex == "man" else self.woman_token
        row_df["p2_token"] = self.man_token if p2_sex == "man" else self.woman_token
        surroundings = self.bg_dict[row_df['surroundings']][self.surrounings_type]
        row_df["surroundings"] = surroundings
        return row_df
    
    def get_id(self, mode, idx):
        df=self.get_row_data(mode, idx)
        id_ = df['id_']
        if self.debug:
            print(f"id: {id_}")
        return  id_

    def get_sex(self, mode, idx):
        df=self.get_row_data(mode, idx)
        p1_sex, p2_sex = df["person1_sex"], df["person2_sex"]
        if self.debug:
            print("pt1 sex: ", p1_sex, " pt2 sex: ", p2_sex)
        return p1_sex, p2_sex

    def get_prompt_replaced(self, mode, prompt_type, idx, with_token = True):
        df=self.get_row_data(mode, idx)
        p1_sex, p2_sex = df["person1_sex"], df["person2_sex"]
        head, layout, expression, surroundings = df["head"], df["layout"], df["expression"], df["surroundings"]
        if with_token:
            action_replaced= self.replace_prompt(df["action"],df["p1_token"],df["p2_token"])
        else:
            action_replaced = self.replace_prompt(df["action"],f"a {p1_sex}",f"a {p2_sex}")

        if  prompt_type== "simple":
            prompt = f"A photo of {action_replaced}"

        elif prompt_type == "action+layout":
            if layout != "":
                prompt = f"{head} {action_replaced}, {layout}"
            else:
                prompt = f"{head} {action_replaced}"

        elif prompt_type == "action+expression":
            prompt = f"A photo of {action_replaced}, {expression}"

        elif prompt_type == "action+background":
            prompt = f"A photo of {action_replaced}, {surroundings}"

        elif prompt_type == "all":
            if layout != "":
                prompt = f"{head} {action_replaced}, {layout}, {expression}, {surroundings}"
            else:
                prompt = f"{head} {action_replaced}, {expression}, {surroundings}"
        else:
            raise ValueError("promtp_tpye is not appropriate. Chose from [simple, action+layout, action+expression, action+background, all]")
        prompt = prompt + ", Ultra HD quality."
        prompt = self.replace_prompt(prompt, f"the {p1_sex}", f"the {p2_sex}")
        if self.debug:
            if with_token:
                print(f"prompt replaced with token:{prompt}")
            else:
                print(f"prompt replaced with class:{prompt}")
        return prompt
    
    def get_prompt_indivisual(self, mode, prompt_type, idx):
        df=self.get_row_data(mode, idx)
        p1_sex, p2_sex = df["person1_sex"], df["person2_sex"]
        p1_expression, p2_expression = df["person1_expression"], df["person2_expression"]
        p1_action, p2_action = df["person1_action"], df["person2_action"]
        p1_token, p2_token = df["p1_token"], df["p2_token"]
        head = df["head"]

        p1_action_replaced_token = self.replace_prompt(p1_action,p1_token, f"a {p2_sex}")
        p2_action_replaced_token = self.replace_prompt(p2_action,f"a {p1_sex}",p2_token)

        if  prompt_type== "simple":
            p1_prompt = f"A photo of {p1_action_replaced_token}."
            p2_prompt = f"A photo of {p2_action_replaced_token}."

        elif prompt_type == "action+layout":      
            p1_prompt = f"{head} {p1_action_replaced_token}."
            p2_prompt = f"{head} {p2_action_replaced_token}."

        elif prompt_type == "action+expression":
            p1_prompt = f"A photo of {p1_action_replaced_token}, {p1_expression}."
            p2_prompt = f"A photo of {p2_action_replaced_token}, {p2_expression}."

        elif prompt_type == "action+background":
            p1_prompt = f"A photo of {p1_action_replaced_token}."
            p2_prompt = f"A photo of {p2_action_replaced_token}."

        elif prompt_type == "all":
            p1_prompt = f"{head} {p1_action_replaced_token}, {p1_expression}."
            p2_prompt = f"{head} {p2_action_replaced_token}, {p2_expression}."
        else:
            raise ValueError("promtp_tpye is not appropriate. Chose from [simple, action+layout, action+expression, action+background, all]")
    

        p1_prompt = self.replace_prompt(p1_prompt, f"the {p1_sex}", f"the {p2_sex}")
        p2_prompt = self.replace_prompt(p2_prompt, f"the {p1_sex}", f"the {p2_sex}")

        if mode == "easy":
            p2_prompt = ""

        if self.debug:
            print("pt1: ", p1_prompt)
            print("pt2: ", p2_prompt)
        return p1_prompt, p2_prompt
    
    def get_idx_info(self, mode, prompt_type, idx):
        id_ = self.get_id(mode, idx)
        prompt_token = self.get_prompt_replaced(
            mode, prompt_type, idx, with_token=True
        )
        prompt_class = self.get_prompt_replaced(
            mode, prompt_type, idx, with_token=False
        )
        pt1, pt2 = self.get_prompt_indivisual(
            mode, prompt_type, idx
        )
        p1_sex, p2_sex = self.get_sex(mode, idx)

        data = {
            "id_": id_,
            "prompt_token": prompt_token,
            "prompt_class": prompt_class,
            "pt1": pt1,
            "pt2": pt2,
            "p1_sex": p1_sex,
            "p2_sex": p2_sex,
        }
        return data
    
def load_yaml_config(yaml_path: str = "config.yaml") -> dict:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for key in config.keys():
            val = config[key]
            print(f"{key}: {val}")
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
        '--csv_path', type=str, default="CC-AlignBench/data.csv")
    parser.add_argument(
        '--bg_path', type=str, default="CC-AlignBench/background.json")
    parser.add_argument(
        '--surrounings_type', type=str, default="simple", choices=["simple", "detail"])
    parser.add_argument(
        '--man_token', type=str, default="a man")
    parser.add_argument(
        '--woman_token', type=str, default="a woman")
    parser.add_argument(
        '--mode', type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument(
        '--prompt_type', type=str, default="all", choices=["simple", "action+layout", "action+expression", "action+background", "all"])
    parser.add_argument(
        '--index_list', type=int, nargs='+')
    parser.add_argument(
        '--debug', action='store_true')
    args = parser.parse_args()

    dataloader = DataLoader(
        csv_path = args.csv_path, 
        bg_path = args.bg_path, 
        surrounings_type = args.surrounings_type, 
        man_token = args.man_token, 
        woman_token = args.woman_token,
        debug = args.debug)

    index_list = range(dataloader.get_len_of_data(args.mode)) if args.index_list == None else args.index_list
    for idx in index_list:
        data = dataloader.get_idx_info(args.mode, args.prompt_type, idx)
        print(data)