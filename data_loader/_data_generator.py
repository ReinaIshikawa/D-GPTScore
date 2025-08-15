import argparse
import csv
from _prompt_loader import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path', type=str, default="CC-AlignBench/data.csv")
    parser.add_argument(
        '--bg_path', type=str, default="CC-AlignBench/background.json")
    parser.add_argument(
        '--surrounings_type', type=str, default="simple", choices=["simple", "detail"])
    parser.add_argument(
        '--debug', action='store_true')
    parser.add_argument(
        '--output_csv', type=str, default="./CC-AlignBench/cc-alignbench-data.csv")
    args = parser.parse_args()

    dataloader = DataLoader(
        csv_path = args.csv_path, 
        bg_path = args.bg_path, 
        surrounings_type = args.surrounings_type, 
        man_token = "<|man_1|>", 
        woman_token = "<|woman_1|>",
        debug = args.debug)

    # CSVファイルに書き込むための準備
    csv_data = []
    
    # mode名のマッピング
    # mode_mapping = {
    #     "simple": "act",
    #     "action+layout": "act+lyt", 
    #     "action+expression": "act+exp",
    #     "action+background": "act+sur",
    #     "all": "all"
    # }

    for mode in ["easy", "medium", "hard"]:
        for prompt_type in ["all", "action+layout", "action+expression", "action+background", "simple"]:
            for idx in range(dataloader.get_len_of_data(mode)):
                data = dataloader.get_idx_info(mode, prompt_type, idx)
                
                # CSV用のデータを準備
                csv_row = {
                    'mode': mode,
                    'prompt_type': prompt_type,
                    'id': data["id_"],
                    'prompt': data["prompt_token"],
                    'p1_sex': data["p1_sex"],
                    'p2_sex': data["p2_sex"],
                    'p1_prompt': data["pt1"],
                    'p2_prompt': data["pt2"]
                }
                csv_data.append(csv_row)

    # CSVファイルに保存
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['mode', 'prompt_type', 'id', 'prompt', 'p1_sex', 'p2_sex', 'p1_prompt', 'p2_prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"saved as {args.output_csv}")







