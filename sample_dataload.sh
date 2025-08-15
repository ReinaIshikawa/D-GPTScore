python data_loader/prompt_loader.py\
 --csv_path ./CC-AlignBench/cc-alignbench-data.csv \
 --man_token "a man" \
 --woman_token "a woman" \
 --index_list 0 9 11 23 34 \
 --prompt_type all \
 --mode hard


python data_loader/prompt_loader.py \
 --yaml_path ./eval_config.yaml \
 --man_token "A*" \
 --woman_token "B*" \
 --index_list 46 \
 --prompt_type action+expression \
 --mode easy