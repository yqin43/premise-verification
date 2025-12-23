import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
with open('../KG-FPQ-data/art_YN.json', 'r') as file:
    json_data = json.load(file)

T_q = []
F_q = []
for data in json_data:
    T_q.append(data['TPQ'])
    for key, value in data.items():
        if key.startswith("FPQ_"):
            F_q.append(value)

all_q = T_q + F_q

os.environ['HF_TOKEN'] = "" # Set your Hugging Face token here

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

##### original #####
responses = []
for q in all_q:
    qq = q + ' This question may contain a fasle premise.'
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, do_sample=False, pad_token_id=tokenizer.eos_token_id,max_new_tokens=100)
 
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response[len(q):] if response.startswith(q) else response
    responses.append(response.strip())

# with open('mistral_orig_ans.json', 'w') as file:
#     json.dump(responses, file)

with open('mistral_prompt.json', 'w') as file:
    json.dump(responses, file)
# import json
# with open('mistral_orig_ans.json', 'r') as file:
#     responses = json.load(file)
    
t_correct = 0
t_incorrect = 0
for r in responses:
    if r[0] == 'Y' or ('yes' in r.lower() and r[0] != 'N'):
        t_correct+=1
    if r[0] == 'N' or ('no' in r.lower() and r[0] != 'Y' and 'yes' not in r.lower()):
        t_incorrect+=1

f_correct = 0
f_incorrect = 0
for i, r in enumerate(responses):
    if r[0] == 'Y' or ('yes' in r.lower() and r[0] != 'N'):
        f_incorrect+=1
    if r[0] == 'N' or ('no' in r.lower() and r[0] != 'Y' and 'yes' not in r.lower()):
        f_correct+=1

print(f"true premise: correct: {t_correct}, incorrect: {t_incorrect}")
print(f"false premise: correct: {f_correct}, incorrect: {f_incorrect}")

#### detect #####
import json
with open('./g_retri_gg.json', 'r') as file:
    detect = json.load(file)

# prompt = "Give me a short introduction to large language model."
tp = ' Note this question contains false premise.'

responses = []

for i in range(len(all_q)):
    if 'Yes' in detect[i]:
        qq = all_q[i] + tp
    else:
        qq = all_q[i]
    inputs = tokenizer(qq, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, do_sample=False, pad_token_id=tokenizer.eos_token_id,max_new_tokens=100)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response[len(q):] if response.startswith(q) else response
    
    responses.append(response.strip())



with open('mistral_detect_ans.json', 'w') as file:
    json.dump(responses, file)


print('mistral after detect ANS')

# import json
# with open('mistral_detect_ans.json', 'r') as file:
#     responses = json.load(file)
    
t_correct = 0
t_incorrect = 0
for r in responses:
    if r[0] == 'Y' or ('yes' in r.lower() and r[0] != 'N'):
        t_correct+=1
    if r[0] == 'N' or ('no' in r.lower() and r[0] != 'Y' and 'yes' not in r.lower()):
        t_incorrect+=1

f_correct = 0
f_incorrect = 0
for i, r in enumerate(responses):
    if r[0] == 'Y' or ('yes' in r.lower() and r[0] != 'N'):
        f_incorrect+=1
    if r[0] == 'N' or ('no' in r.lower() and r[0] != 'Y' and 'yes' not in r.lower()):
        f_correct+=1

print(f"true premise: correct: {t_correct}, incorrect: {t_incorrect}")
print(f"false premise: correct: {f_correct}, incorrect: {f_incorrect}")