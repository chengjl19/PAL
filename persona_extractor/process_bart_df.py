import json
from tqdm import tqdm
import torch
import random
from transformers import BartTokenizer
import pandas as pd

datatype = ['test', 'train', 'valid']
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
for name in datatype:
    file_path = f'Persona_chat/personachat/{name}_both_original.txt'
    with open(file_path) as f:
        lines = f.readlines()
    tmp_data = []
    i = 0
    pbar = tqdm(total=len(lines))
    while i < len(lines):
        count = 0
        persona_a = []
        persona_b = []
        context_a = []
        context_b = []
        while lines[i].count('your persona:'):
            persona_a.append(lines[i][lines[i].find(':')+2:-1])
            i += 1
            count += 1
        while lines[i].count("partner's persona:"):
            persona_b.append(lines[i][lines[i].find(':')+2:-1])
            i += 1
            count += 1
        while i < len(lines) and not lines[i].count("your persona:"):
            sen = lines[i].split('\t')
            context_b.append(sen[0][sen[0].find(' ')+1:])
            context_a.append(sen[1])
            i += 1
            count += 1
        tmp_data.append({
            'persona': (" " + "<persona>" + " ").join(persona_a) + " " + tokenizer.eos_token,
            'context': (" " + "<sep>" + " ").join(context_a) + " " + tokenizer.eos_token
        })
        tmp_data.append({
            'persona': (" " + "<persona>" + " ").join(persona_b) + " " + tokenizer.eos_token,
            'context': (" " + "<sep>" + " ").join(context_b) + " " + tokenizer.eos_token
        })
        pbar.update(count)
    pbar.close()
    tmp_df = pd.DataFrame({
        'context': [i['context'] for i in tmp_data],
        'persona': [i['persona'] for i in tmp_data]
    })
    tmp_df.to_csv(f"{name}.csv")
