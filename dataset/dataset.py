import pandas as pd
import torch
import random
import time
from pathlib import Path

from datetime import timedelta
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, ChineseCLIPProcessor, AutoImageProcessor, ViTFeatureExtractor
from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def convert_onehot(config, label):
    onehot_label = [0 for i in range(config.num_classes)]
    onehot_label[int(label)] = 1
    return onehot_label


def read_data(config, training=True):
    # data = None
    if training:
        ext = Path(config.train_path).suffix.lower()
        if ext in [".json"]:
            data = pd.read_json(config.train_path)
        else:
            data = pd.read_csv(config.train_path)
    else:
        ext = Path(config.test_path).suffix.lower()
        if ext in [".json"]:
            data = pd.read_json(config.test_path)
        else:
            data = pd.read_csv(config.test_path)
    return data


# 适合 bert、roberta 和 t5
def brief_prompt(config, data_row, tokenizer, special_token_mapping):

    text = data_row.text + ' . '
    if config.if_entity:
        text += data_row.entity + ' . '
    if config.if_race:
        text += data_row.race + ' . '
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = data_row.caption + ' . '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    if config.text_encoder == "roberta-base" or config.text_encoder == "roberta-large":
        prompt = " It is <mask> ."  # length = 4
    elif config.text_encoder == "bert-base-uncased":
        prompt = " It is [MASK] ."  # length = 4
    elif config.text_encoder == "flan-t5-base":
        prompt = " It is [mask] ."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # get complete prompt
    if config.text_encoder == "flan-t5-base":
        if len(original_ids) > config.pad_size - len(caption_ids) - len(prompt_ids) - 1:
            original_ids = original_ids[:config.pad_size -
                                        len(caption_ids) - len(prompt_ids) - 1]
        prompt_input_ids = caption_ids + original_ids + prompt_ids + \
            [special_token_mapping['[SEP]']]  # input_ids
    else:
        if len(original_ids) > config.pad_size - len(caption_ids) - len(prompt_ids) - 2:
            original_ids = original_ids[:config.pad_size -
                                        len(caption_ids) - len(prompt_ids) - 2]
        prompt_input_ids = [special_token_mapping['[CLS]']] + caption_ids + \
            original_ids + prompt_ids + \
            [special_token_mapping['[SEP]']]  # input_ids

    return prompt_input_ids


# For T5
def template_1(config, data_row, tokenizer):

    text = 'Text:' + data_row.text + ' . '
    if config.if_entity:
        text += 'Entity:' + data_row.entity + ' . '
    if config.if_race:
        text += 'Race:' + data_row.race + ' . '
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = 'Caption:' + data_row.caption + ' . '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    prompt = " It is :"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return original_ids, caption_ids, prompt_ids


def template_2(config, data_row, tokenizer):

    text = 'Text:' + data_row.text + ' . '
    if config.if_entity:
        text += 'Keyword:' + data_row.entity + ' . '
    if config.if_race:
        text += data_row.race + ' . '
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = 'Caption:' + data_row.caption + ' . '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    prompt = " It is [mask]."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return original_ids, caption_ids, prompt_ids


def template_3(config, data_row, tokenizer):

    text = 'Text:' + data_row.text + ' . '
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = 'Caption:' + data_row.caption[:-1] + ' . '
    if config.if_entity:
        caption += 'Entity:' + data_row.entity + ' . '
    if config.if_race:
        caption += 'Race:' + data_row.race + ' . '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    prompt = " It is [mask]."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return original_ids, caption_ids, prompt_ids


def template_4(config, data_row, tokenizer):

    text = 'Text:' + data_row.text + ' . '
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = 'Caption:' + data_row.caption[:-1] + ' . '
    if config.if_entity:
        caption += 'Keyword:' + data_row.entity + ' . '
    if config.if_race:
        caption += data_row.race + ' . '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    prompt = " It is [mask]."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return original_ids, caption_ids, prompt_ids


def template_5(config, data_row, tokenizer):

    text = f'Given the sentence "{data_row.text}", '
    if config.if_entity:
        text += f'and related keywords "{data_row.entity}", '
    if config.if_race:
        text += f'"{data_row.race}".'
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = f'The caption is "{data_row.caption[:-1]}". '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    prompt = "Based on the common sense, determine if it is good or bad."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return original_ids, caption_ids, prompt_ids


def template_8(config, data_row, tokenizer):

    text = f'Given the sentence "{data_row.text}", '
    if config.if_entity:
        text += f'and related keywords "{data_row.entity}", '
    if config.if_race:
        text += f'"{data_row.race}".'
    original_ids = tokenizer.encode(
        text, add_special_tokens=False)  # list

    caption = f'The caption is "{data_row.caption[:-1]}". '
    caption_ids = tokenizer.encode(caption, add_special_tokens=False)

    prompt = "Based on the common sense, it is [mask]."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return original_ids, caption_ids, prompt_ids

# For T5
def designed_prompt(config, data_row, tokenizer, special_token_mapping):

    if config.template == 1:
        original_ids, caption_ids, prompt_ids = template_1(config, data_row, tokenizer)
    elif config.template == 2:
        original_ids, caption_ids, prompt_ids = template_2(config, data_row, tokenizer) 
    if config.template == 3:
        original_ids, caption_ids, prompt_ids = template_3(config, data_row, tokenizer)
    elif config.template == 4:
        original_ids, caption_ids, prompt_ids = template_4(config, data_row, tokenizer) 
    elif config.template == 5:
        original_ids, caption_ids, prompt_ids = template_5(config, data_row, tokenizer) 
    elif config.template == 8:
        original_ids, caption_ids, prompt_ids = template_8(config, data_row, tokenizer) 

    if config.template == 1 or config.template == 2 or config.template == 5 or config.template == 6 or config.template == 7 or config.template == 8: 
        if len(original_ids) > config.pad_size - len(caption_ids) - len(prompt_ids) - 1:
            original_ids = original_ids[:config.pad_size -
                                        len(caption_ids) - len(prompt_ids) - 1]
        prompt_input_ids = caption_ids + original_ids + prompt_ids + \
            [special_token_mapping['[SEP]']]  # input_ids
    
    if config.template == 3 or config.template == 4: 
        if len(caption_ids) > config.pad_size - len(original_ids) - len(prompt_ids) - 1:
            caption_ids = caption_ids[:config.pad_size -
                                        len(original_ids) - len(prompt_ids) - 1]
        prompt_input_ids = original_ids + caption_ids + prompt_ids + \
            [special_token_mapping['[SEP]']]  # input_ids
        
    return prompt_input_ids


class MemeDataset(Dataset):

    def __init__(self, config, training=True):
        self.config = config
        self.model_name = config.model_name
        self.data = read_data(config, training)
        self.max_len = config.pad_size

        if config.text_encoder == "clip":
            self.tokenizer = ChineseCLIPProcessor.from_pretrained(
                config.chinese_clip_path)
        elif config.text_encoder == "chinese-roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained(
                config.chinese_roberta_path)
        elif config.text_encoder == "roberta-base":
            self.tokenizer = RobertaTokenizer.from_pretrained(
                config.roberta_base_path)
        elif config.text_encoder == "roberta-large":
            self.tokenizer = RobertaTokenizer.from_pretrained(
                config.roberta_large_path)
        elif config.text_encoder == "flan-t5-base":
            self.tokenizer = T5Tokenizer.from_pretrained(
                config.flan_t5_base_path)

        if config.image_encoder == "clip":
            self.extractor = ChineseCLIPProcessor.from_pretrained(
                config.chinese_clip_path)
        elif config.image_encoder == "resnet":
            self.extractor = AutoImageProcessor.from_pretrained(
                config.resnet_path)
        elif config.image_encoder == "vit":
            self.extractor = AutoImageProcessor.from_pretrained(
                config.vit_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row.text
        label = data_row.label
        file_name = data_row.path

        label = torch.tensor(convert_onehot(self.config, label)).float()
        image = Image.open(self.config.meme_path + file_name).convert('RGB')

        # For text
        text_inputs = self.tokenizer(
            text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        # For image
        image_inputs = self.extractor(images=image, return_tensors='pt')

        input_ids = text_inputs["input_ids"].squeeze()
        attention_mask = text_inputs["attention_mask"].squeeze()

        image_tensor = image_inputs["pixel_values"].squeeze()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_tensor=image_tensor,
            label=label,
        )


class PromptMemeDataset(MemeDataset):

    def __init__(self, config, training=True):
        super().__init__(config, training)
        self.label_list = config.label_list
        self.text_encoder = config.text_encoder

        if self.text_encoder == "roberta-base" or self.text_encoder == "roberta-large":
            self.special_token_mapping = {
                '[CLS]': self.tokenizer.convert_tokens_to_ids('<s>'),
                '[MASK]': self.tokenizer.mask_token_id,
                '[PAD]': self.tokenizer.pad_token_id,  # 1 for roberta
                '[SEP]': self.tokenizer.convert_tokens_to_ids('</s>')
            }
        elif self.text_encoder == "bert-base-uncased":
            self.special_token_mapping = {
                '[CLS]': self.tokenizer.convert_tokens_to_ids('[CLS]'),
                '[MASK]': self.tokenizer.mask_token_id,
                '[PAD]': self.tokenizer.pad_token_id,  # 1 for roberta
                '[SEP]': self.tokenizer.convert_tokens_to_ids('[SEP]')
            }

    def __getitem__(self, index):
        dict = super().__getitem__(index)

        data_row = self.data.iloc[index]

        prompt_input_ids = brief_prompt(
            self.config, data_row, self.tokenizer, self.special_token_mapping)
        prompt_attention_mask = [1 for i in range(len(prompt_input_ids))]

        while len(prompt_input_ids) < self.max_len:
            prompt_input_ids.append(self.special_token_mapping['[PAD]'])
            prompt_attention_mask.append(0)

        mask_pos = [prompt_input_ids.index(
            self.special_token_mapping['[MASK]'])]

        dict["prompt_input_ids"] = torch.tensor(prompt_input_ids)
        dict["prompt_attention_mask"] = torch.tensor(prompt_attention_mask)
        dict["mask_pos"] = torch.tensor(mask_pos)

        return dict


class PromptMemeT5Dataset(MemeDataset):
    def __init__(self, config, training=True):
        super().__init__(config, training)
        self.label_list = config.label_list
        self.text_encoder = config.text_encoder

        if self.text_encoder == "flan-t5-base":
            self.special_token_mapping = {
                '[PAD]': self.tokenizer.pad_token_id,
                '[SEP]': self.tokenizer.convert_tokens_to_ids('</s>')
            }

    def __getitem__(self, index):
        dict = super().__getitem__(index)

        data_row = self.data.iloc[index]

        if self.config.template == 0:
            prompt_input_ids = brief_prompt(
                self.config, data_row, self.tokenizer, self.special_token_mapping)
        else:
            prompt_input_ids = designed_prompt(
                self.config, data_row, self.tokenizer, self.special_token_mapping)
            
        prompt_attention_mask = [1 for i in range(len(prompt_input_ids))]

        while len(prompt_input_ids) < self.max_len:
            prompt_input_ids.append(self.special_token_mapping['[PAD]'])
            prompt_attention_mask.append(0)

        output = self.tokenizer.encode_plus(
            self.config.label_list[data_row.label], add_special_tokens=True, max_length=self.config.label_pad_size, padding="max_length")

        dict["prompt_input_ids"] = torch.tensor(prompt_input_ids)
        dict["prompt_attention_mask"] = torch.tensor(prompt_attention_mask)
        dict["output_ids"] = torch.tensor(output.input_ids)
        dict["output_mask"] = torch.tensor(output.attention_mask)

        return dict
