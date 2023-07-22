# PAL

Code and data for ACL 2023 findings: **PAL: Persona-Augmented Emotional Support Conversation Generation** (https://arxiv.org/abs/2212.09235)

If you use our codes or your research is related to our paper, please kindly cite our paper:

```bib
@inproceedings{cheng-etal-2023-pal,
    title = "{PAL}: Persona-Augmented Emotional Support Conversation Generation",
    author = "Cheng, Jiale  and
      Sabour, Sahand  and
      Sun, Hao  and
      Chen, Zhuang  and
      Huang, Minlie",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.34",
    pages = "535--554",
    abstract = "Due to the lack of human resources for mental health support, there is an increasing demand for employing conversational agents for support. Recent work has demonstrated the effectiveness of dialogue models in providing emotional support. As previous studies have demonstrated that seekers{'} persona is an important factor for effective support, we investigate whether there are benefits to modeling such information in dialogue models for support. In this paper, our empirical analysis verifies that persona has an important impact on emotional support. Therefore, we propose a framework for dynamically inferring and modeling seekers{'} persona. We first train a model for inferring the seeker{'}s persona from the conversation history. Accordingly, we propose PAL, a model that leverages persona information and, in conjunction with our strategy-based controllable generation method, provides personalized emotional support. Automatic and manual evaluations demonstrate that PAL achieves state-of-the-art results, outperforming the baselines on the studied benchmark. Our code and data are publicly available at https://github.com/chengjl19/PAL.",
}

## Persona Extractor
Our persona extractor is trained using the dataset, PersonaChat, see our paper for details on how to do this. <br> 
The training code is located in the `persona_extractor` path, where `process_bart_df` is the training data preprocessing code and `train_bart` is the training code.

To run this code, just put the PersonaChat dataset under this path and
```python
python process_bart_df.py
python train_bart.py
```

## PESC Dataset
Using the trained persona extractor, we can get the seeker's persona from ESC dataset.

In PESC dataset, we additionally add two fields: `persona` and `persona_list`.

`persona` contains the persona inferred from the whole conversation.
`persona_list` contains the persona inferred from each turn of the seeker's utterance after 3.

## PAL Model

Our code (in `codes`) mainly references https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes_zcj

### Training

1.data process

```python
python codes/_reformat/process.py --add_persona True
```

2.model training

You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) model and put the `pytorch_model.bin` file in `Blenderbot_small-90M`.

Then run `RUN/prepare_strat.sh`

And run `RUN/train_strat.sh`

3.model inference

run `RUN/infer_strat.sh`

4.model interact

run `RUN/interact_strat.sh`