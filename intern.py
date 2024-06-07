import json
import time
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "/home/yerong2/models/internlm-xcomposer2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')


model = model.eval()

# Load the data from the JSON file
with open('data/testmini.json', 'r') as file:
    data = json.load(file)

# Iterate through each dictionary in the list
# cnt = 0
for key in tqdm(data):
    entry = data[key]

    # Extract the relevant fields from each entry
    query_cot = entry["question"]
    image_path = entry["image"]
    image = 'data/' + image_path
    
    
    # Define the query for the model
    query = '<ImageHere>' + query_cot

    # Generate the response from the model
    with torch.cuda.amp.autocast():
        response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)

    # Add the response to the entry
    entry["model_answer"] = response
    # cnt+= 1
    # if cnt > 3: break

# Save the updated data to a new JSON file
with open('data/testmini_ans.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Responses saved to data/testmini_ans.json")

