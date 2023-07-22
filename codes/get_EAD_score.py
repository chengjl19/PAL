import pandas as pd
import nltk
from nltk import ngrams, sent_tokenize
import tqdm
import json


def cal_EAD(sents, n=4):
    all_ngrams = []
    for i in tqdm.trange(len(sents)):
        words = nltk.word_tokenize(sents[i])
        grams = ngrams(words, n)
        all_ngrams.extend(grams)
    N = len(set(all_ngrams))
    C = len(all_ngrams)
    V = 54956
    return N / (V*(1-(((V-1)/V)**C)))


if __name__ == '__main__':
    with open('/home/chengjiale/emotion/MISC/generated_data/all_loss/hyp_strategy.json') as f:
        data = json.load(f)
    context = data
    # context = [i['generation'] for i in data]
    sentence = []
    for ctx in context:
        sentence.extend(sent_tokenize(str(ctx)))
    print(cal_EAD(sentence, 1))
