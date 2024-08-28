import os

import torch
from transformers import AutoModel, AutoTokenizer

modelpath = os.getenv('MODELS')
model_name_or_path = modelpath + "/internlm-xcomposer2-vl-7b"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

query = '<ImageHere>Please describe this image in detail.'
image = './examples/cars1.jpg'
with torch.cuda.amp.autocast():
  response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
print(response)
#The image features a quote by Oscar Wilde, "Live life with no excuses, travel with no regret,"
# set against a backdrop of a breathtaking sunset. The sky is painted in hues of pink and orange,
# creating a serene atmosphere. Two silhouetted figures stand on a cliff, overlooking the horizon.
# They appear to be hiking or exploring, embodying the essence of the quote.
# The overall scene conveys a sense of adventure and freedom, encouraging viewers to embrace life without hesitation or regrets.

