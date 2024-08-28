import json
import time
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
import torch
import os
from tqdm import tqdm
modelpath = os.getenv('MODELS')

# Load the model and tokenizer
model_name_or_path = modelpath + "/internlm-xcomposer2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).half().eval().cuda()
model.tokenizer = tokenizer

# Load the data from the JSON file
FILE = 'testmini'
with open(f'data/{FILE}.json', 'r') as file:
    data = json.load(file)

# Iterate through each entry in the dataset
cnt = 0
for key in tqdm(data):
    entry = data[key]
    
    # Construct the query with the question
    query_cot = f'Question: {entry["question"]}'
    image_path = entry["image"]
    image = ['data/' + image_path]
    
    # Define the query for the model
    query = '<ImageHere>' + query_cot

    query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
    image = ['../representation-engineering/examples/cars1.jpg',
            '../representation-engineering/examples/cars2.jpg',
            '../representation-engineering/examples/cars3.jpg',]
    
    # Generate the response from the model
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
    
    # Update the entry with the query and model's response
    entry["query"] = query_cot
    entry["model_answer"] = response
    cnt += 1
    break
    # Stop after processing 3 entries
    if cnt > 3: 
        break

# Define the output file name
OUTPUTFILE = f'data/{FILE}_{model_name_or_path.split("/")[-1]}_ans.json'

# Save the updated data to a new JSON file
with open(OUTPUTFILE, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Responses saved to {OUTPUTFILE}")
