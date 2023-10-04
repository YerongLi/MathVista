cd ../evaluation

#### gpt4_2shot_solution #####
# generate solution
python generate_response.py \
--model gpt-4-0613 \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution.json \
--shot_num 2 \
--shot_type solution

# extract answer
python extract_answer.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution.json

calculate score
python calculate_score.py \
--output_dir ../results/gpt4 \
--output_file output_gpt4_2shot_solution.json \
--score_file scores_gpt4_2shot_solution.json
