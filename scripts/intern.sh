cd ../evaluation

##### llava-llama-2-13b #####
# Extract Answer
python extract_answer.py \
--output_file ../data/testmini_ans.json \
--response_label model_answer
