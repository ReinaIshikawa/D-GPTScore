import json
import os

# 各フォルダのcorr.jsonを読み込む
folders = {
    'GPT_ours_ave': 'Ours (average)',
    'GPT_ours_ave++': 'Ours++ (average)',
    'GPT_ours_linear': 'Ours (linear regression)',
    'GPT_ours_linear++': 'Ours++ (linear regression)',
    'GPT4omini_ours_ave': 'Ours (GPT-4o-mini)'
}

data = {}
for folder, display_name in folders.items():
    corr_file = os.path.join(folder, 'corr.json')
    if os.path.exists(corr_file):
        with open(corr_file, 'r') as f:
            data[folder] = json.load(f)

# 各手法について相関係数を出力
for folder, display_name in folders.items():
    if folder in data:
        # 指定された順序で相関係数を取得
        pearson_01 = data[folder]['01_CustomDiffusion']['pearson_corr']
        spearman_01 = data[folder]['01_CustomDiffusion']['spearman_corr']
        
        pearson_02 = data[folder]['02_OMG_lora']['pearson_corr']
        spearman_02 = data[folder]['02_OMG_lora']['spearman_corr']
        
        pearson_03 = data[folder]['03_OMG_instantID']['pearson_corr']
        spearman_03 = data[folder]['03_OMG_instantID']['spearman_corr']
        
        pearson_04 = data[folder]['04_fastcomposer']['pearson_corr']
        spearman_04 = data[folder]['04_fastcomposer']['spearman_corr']
        
        pearson_05 = data[folder]['05_Mix-of-Show']['pearson_corr']
        spearman_05 = data[folder]['05_Mix-of-Show']['spearman_corr']
        
        pearson_06 = data[folder]['06_DreamBooth']['pearson_corr']
        spearman_06 = data[folder]['06_DreamBooth']['spearman_corr']
        
        pearson = data[folder]['total_pearson_corr']
        spearman = data[folder]['total_spearman_corr']
        
        # 出力形式に合わせて整形（pearson / spearmanの順）
        output = f"{display_name} & {pearson_01:.4f} / {spearman_01:.4f} & {pearson_02:.4f} / {spearman_02:.4f} & {pearson_03:.4f} / {spearman_03:.4f} & {pearson_04:.4f} / {spearman_04:.4f} & {pearson_05:.4f} / {spearman_05:.4f} & {pearson_06:.4f} / {spearman_06:.4f} & {pearson:.4f} / {spearman:.4f} \\\\"
        
        print(output)
