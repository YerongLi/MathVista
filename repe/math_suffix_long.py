import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch


# Set up the argument parser
def main():
    parser = argparse.ArgumentParser(description='Process checkpoint path.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--ablation', action='store_true', default=False, help='Enable ablation mode')
    args = parser.parse_args()
    
    ablation = args.ablation
    datasets_dir = os.getenv("DATASETS")
    image_path_prefix = f"{datasets_dir}/MathVista/"
    dataset = load_dataset("AI4Math/MathVista")
    # /home/yerong2/representation-engineering/lorra_finetune/llavafine/output/internlm-xcomposer2-7b-chat/v130-20241026-015150/checkpoint-700
    # Parse the arguments

    # Extract the checkpoint path
    checkpoint_path = args.checkpoint

    # Check the length of the split path
    checkpoint_parts = checkpoint_path.split('/')
    
    # Set a threshold for the length of the path
    threshold_length = 3  # You can adjust this value as needed
    if len(checkpoint_parts) < threshold_length:
        MODEL_TYPE = checkpoint_path.replace('-', '_')
        filename = 'ablation-suffix.json' if ablation else 'original.json'
    else:
        # Extract the model type
        MODEL_TYPE = checkpoint_parts[-3].replace('-', '_')
        filename = os.path.basename(os.path.dirname(checkpoint_path)) + '.json'

    model_type = getattr(ModelType, MODEL_TYPE)
    model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                           model_kwargs={'device_map': 'auto'})

    if len(checkpoint_parts) >= threshold_length:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        print(f" ==== Model merged successfully from checkpoint: {checkpoint_path}")
    template_type = get_default_template_type(model_type)


    model.generation_config.max_new_tokens = 2048
    template = get_template(template_type, tokenizer)
    if ablation:
        pass
        # template.default_system = 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.'
    suffix = 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.'
    print(template.default_system)
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
    cnt = 0
    filename = f'{save_prefix}/{filename}' if checkpoint is None else 'original.json'
    for entry in tqdm(dataset["testmini"]):
        if not (cnt % 100): print(filename)
        cnt+= 1
        # Create query string
        query = f"<img>{image_path_prefix+entry['image']}</img>\n{entry['query']}?{suffix}"
        
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
            'model': str(model_type)
        }
    
    # Write to file with proper formatting
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Results successfully dumped to {filename}")
if __name__ == '__main__':
    main()