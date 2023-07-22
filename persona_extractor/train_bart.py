import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import top_k_top_p_filtering
import math
import random
import re
import argparse
from tqdm import trange


class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, total_steps=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        self.test_loss = []
        self.total_steps = total_steps
        # self.freeze_encoder = freeze_encoder
        # self.freeze_embeds_ = freeze_embeds
        print(hparams)

        if hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())

        if hparams.freeze_embeds:
            self.freeze_embeds()

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, batch):
        batch = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }
        # print(batch['labels'][0])
        return self.model(**batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=hparams.warmup_steps, num_training_steps=self.total_steps
        )
        return [optimizer], [{"scheduler": scheduler}]

    def training_step(self, batch, batch_idx):

        # print(batch)
        outputs = self.forward(batch)
        loss = outputs[0]

        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch[0].size())
        outputs = self.forward(batch)
        loss = outputs[0]

        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):

        outputs = self.forward(batch)
        loss = outputs[0]
        self.test_loss.append(loss)
        return loss

    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)
        print("average test loss: {}".format(sum(self.test_loss) / len(self.test_loss)))

    # Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, text, eval_beams, early_stopping=True, max_len=128):
        ''' Function to generate text '''
        # print(text)
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.tokenizer.eos_token_id,
            num_beams=eval_beams,
            max_length=max_len,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=early_stopping
        )
        return [self.tokenizer.decode(w, clean_up_tokenization_spaces=True)
                for w in generated_ids]


def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False


# Create a dataloading module as per the PyTorch Lightning Docs
class LitDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.train_data = pd.read_csv(self.data_dir + "/train.csv")
        self.valid_data = pd.read_csv(self.data_dir + "/valid.csv")
        self.test_data = pd.read_csv(self.data_dir + "/test.csv")

    # encode the sentences using the tokenizer
    def setup(self, stage):
        self.train_data = encode_sentences(self.tokenizer, self.train_data['context'], self.train_data['persona'])
        self.valid_data = encode_sentences(self.tokenizer, self.valid_data['context'], self.valid_data['persona'])
        self.test_data = encode_sentences(self.tokenizer, self.test_data['context'], self.test_data['persona'])

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train_data['input_ids'], self.train_data['attention_mask'],
                                self.train_data['labels'])
        train_data = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size, num_workers=hparams.num_workers)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.valid_data['input_ids'], self.valid_data['attention_mask'],
                                self.valid_data['labels'])
        val_data = DataLoader(dataset, batch_size=self.batch_size, num_workers=hparams.num_workers)
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test_data['input_ids'], self.test_data['attention_mask'], self.test_data['labels'])
        test_data = DataLoader(dataset, batch_size=self.batch_size, num_workers=hparams.num_workers)
        return test_data


def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=128, pad_to_max_length=True,
                     return_tensors="pt"):
    ''' Function that tokenizes a sentence
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, labels
    '''

    input_ids = []
    attention_masks = []
    label_ids = []

    for sentence in source_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space=True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space=True
        )
        label_ids.append(encoded_dict['input_ids'] - torch.logical_not(encoded_dict['attention_mask'])*(100+tokenizer.pad_token_id))

    label_ids = torch.cat(label_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": label_ids,
    }
    # print(batch)
    return batch


def main():
    bart_model = BartForConditionalGeneration.from_pretrained(hparams.model_dir_or_name)
    tokenizer = BartTokenizer.from_pretrained(hparams.model_dir_or_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<persona>", "<sep>"]})
    bart_model.resize_token_embeddings(len(tokenizer))
    data = LitDataModule(tokenizer, hparams.train_dir, hparams.train_bath_size)
    tmp_df = pd.read_csv(hparams.train_dir+"/train.csv")
    total_steps = (len(tmp_df) // hparams.train_bath_size + 1) * hparams.max_train_epochs
    model = LitModel(learning_rate=hparams.lr, tokenizer=tokenizer, model=bart_model, total_steps=total_steps)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True)
    earlystop_callback = EarlyStopping(monitor='val_loss',
                                       verbose=True,
                                       patience=5,
                                       mode='min')
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[checkpoint_callback, earlystop_callback],
        max_epochs=hparams.max_train_epochs,
        val_check_interval=0.25,
        min_epochs=1,
        default_root_dir='/home/chengjiale/emotion/Persona_extractor/pl_root'
    )

    trainer.fit(model, data)

    trainer.test(model, datamodule=data)


def infer():
    bart_model = BartForConditionalGeneration.from_pretrained(hparams.model_dir_or_name)
    tokenizer = BartTokenizer.from_pretrained(hparams.model_dir_or_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<persona>", "<sep>"]})
    bart_model.resize_token_embeddings(len(tokenizer))
    model = LitModel.load_from_checkpoint(
        "epoch=7-step=8384.ckpt",
        learning_rate=1e-5, tokenizer=tokenizer, model=bart_model)
    model.to('cuda:0')
    model.eval()
    df = pd.read_csv('test.csv')
    batch_size = hparams.eval_batch_size
    res = []
    start = 0
    context = []
    context_number = []
    dialog_id = []
    tmp_context = df['context'].to_list()
    now_id = 0
    # ESC
    # for i in tmp_context:
    #     ctx_num = 3
    #     for j in eval(i):
    #         context.append(j)
    #         dialog_id.append(now_id)
    #         context_number.append(ctx_num)
    #         ctx_num += 1
    #     now_id += 1
    context = tmp_context
    print(len(context))
    # ESC
    # df = pd.DataFrame({
    #     'context': context,
    #     'dialog_id': dialog_id,
    #     'ctx_num': context_number
    # })
    step = len(context) // batch_size + 1 if len(context) % batch_size else len(context) // batch_size
    for i in trange(step):
        texts = tokenizer(
            context[start:start + batch_size],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_prefix_space=True
        )
        start += batch_size
        texts.to('cuda:0')
        res.extend(model.generate_text(texts, hparams.eval_beams))
    df['infer_res'] = res
    df.to_csv('test_gen_beams_10.csv')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    hparams = argparse.Namespace()
    hparams.num_workers = 16
    hparams.train_bath_size = 16
    hparams.eval_batch_size = 8
    hparams.freeze_encoder = False
    hparams.freeze_embeds = False
    hparams.eval_beams = 10
    hparams.warmup_steps = 100
    hparams.train_dir = '/home/chengjiale/emotion/Persona_extractor/data/both_original'
    hparams.max_train_epochs = 10
    hparams.lr = 1e-5
    hparams.model_dir_or_name = "facebook/bart-large-cnn"
    # training
    main()
    # inference
    infer()