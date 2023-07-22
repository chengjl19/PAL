import json
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
args = parser.parse_args()
input_file = args.input_file
with open(input_file) as f:
    data = json.load(f)
with open(input_file) as f:
    data_p = json.load(f)
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
sum_r = sum_g = 0
for i in trange(len(data)):
    r = data[i]['response']
    p = data_p[i]['persona']
    g = data[i]['generation']
    inputs = tokenizer([r, p, g], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        sim_r_p = 1 - cosine(embeddings[0], embeddings[1])
        sim_g_p = 1 - cosine(embeddings[1], embeddings[2])
        sum_r += sim_r_p
        sum_g += sim_g_p
        data[i]['simcse_score'] = {'response':sim_r_p, 'generation':sim_g_p}
with open(input_file, 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
print("average cos similarity:\n response: {},  generation: {}".format(sum_r/len(data), sum_g/len(data)))
