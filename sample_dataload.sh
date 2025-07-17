python data_loader/prompt_loader.py\
 --csv_path ./CC-AlignBench/data.csv \
 --bg_path ./CC-AlignBench/background.json \
 --man_token "a man" \
 --woman_token "a woman" \
 --index_list 0 9 11 23 34 \
 --debug
python data_loader/prompt_loader.py \
 --csv_path ./CC-AlignBench/data.csv \
 --bg_path ./CC-AlignBench/background.json \
 --man_token "A*" \
 --woman_token "B*" \
 --debug \
 --index_list 46 \
 --prompt_type action+expression \
 --mode easy