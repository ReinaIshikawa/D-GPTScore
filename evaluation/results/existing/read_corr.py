import json

# JSONファイルを読み込む
with open('corr.json', 'r') as f:
    data = json.load(f)

# 生成手法の順序（指定された順番）
methods = ['ArcFace', 'CLIP_TEXT2IMAGE', 'CLIP_TEXT2TEXT', 'CLIP_AESTHETIC', 'DINO', 'wo_r_GPT_vanilla']

# 各生成手法について相関係数を出力
for method in methods:
    if method in data:
        # 指定された順序で相関係数を取得
        pearson_01 = data[method]['pearson_corr_01_CustomDiffusion']
        spearman_01 = data[method]['spearman_corr_01_CustomDiffusion']
        
        pearson_02 = data[method]['pearson_corr_02_OMG_lora']
        spearman_02 = data[method]['spearman_corr_02_OMG_lora']
        
        pearson_03 = data[method]['pearson_corr_03_OMG_instantID']
        spearman_03 = data[method]['spearman_corr_03_OMG_instantID']
        
        pearson_04 = data[method]['pearson_corr_04_fastcomposer']
        spearman_04 = data[method]['spearman_corr_04_fastcomposer']
        
        pearson_05 = data[method]['pearson_corr_05_Mix-of-Show']
        spearman_05 = data[method]['spearman_corr_05_Mix-of-Show']
        
        pearson_06 = data[method]['pearson_corr_06_DreamBooth']
        spearman_06 = data[method]['spearman_corr_06_DreamBooth']
        
        pearson = data[method]['pearson_corr']
        spearman = data[method]['spearman_corr']
        
        # 出力形式に合わせて整形
        if method == 'CLIP_TEXT2IMAGE':
            method_display = 'CLIP T2I'
        elif method == 'CLIP_TEXT2TEXT':
            method_display = 'CLIP T2T'
        elif method == 'CLIP_AESTHETIC':
            method_display = 'CLIP Aes.'
        else:
            method_display = method
            
        # 相関係数を小数点2桁に丸める
        output = f"{method_display} & {pearson_01:.2f} / {spearman_01:.2f} & {pearson_02:.2f} / {spearman_02:.2f} & {pearson_03:.2f} / {spearman_03:.2f} & {pearson_04:.2f} / {spearman_04:.2f} & {pearson_05:.2f} / {spearman_05:.2f} & {pearson_06:.2f} / {spearman_06:.2f} & {pearson:.2f} / {spearman:.2f} \\\\"
        
        print(output)
