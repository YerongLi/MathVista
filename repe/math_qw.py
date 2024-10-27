import os
import json
from datasets import load_dataset
from tqdm import tqdm
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch
datasets_dir = os.getenv("DATASETS")
image_path_prefix = f"{datasets_dir}/MathVista/"
dataset = load_dataset("AI4Math/MathVista")

model_type = ModelType.qwen2_vl_7b_instruct
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
print(f'template_type: {template_type}')

model.generation_config.max_new_tokens = 2048
template = get_template(template_type, tokenizer)
seed_everything(42)
# for i in tqdm(range(len(dataset["testmini"]))):
#     query = f"<img>{image_path_prefix+dataset['testmini'][i]['image']}</img>\n{dataset['testmini'][i]['query']}?"
#     response, history = inference(model, template, query)
result_dict = {}
checkpoint = None
save_prefix = f'../results/{str(model_type)}'
if not os.path.exists(save_prefix):
    os.makedirs(save_prefix)
print(save_prefix)
# exit(0)
ablation = False
filename = 'original.json'

if ablation:
    filename = 'ablation.json'
filename = f'{save_prefix}/{filename}' if checkpoint is None else 'original.json'
for entry in tqdm(dataset["testmini"]):
    # Create query string
    query = f"<img>{image_path_prefix+entry['image']}</img>\n{entry['query']}?"
    
    # Run inference
    try:
        response, history = inference(model, template, query)
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Skipping this iteration.")
        continue
    # Store results
    pid = entry['pid']
    result_dict[pid] = {
        "question": entry["question"],
        "query": entry["question"],
        "image": entry["image"],
        "choices": entry.get("choices"),
        "unit": entry.get("unit"),
        "precision": entry.get("precision"),
        "answer": entry["answer"],
        "question_type": entry["question_type"],
        "answer_type": entry["answer_type"],
        "pid": entry["pid"],
        "metadata": entry["metadata"],
        "response": response,  # Use the inference response
    }

# Write to file with proper formatting
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)

print(f"Results successfully dumped to {filename}")