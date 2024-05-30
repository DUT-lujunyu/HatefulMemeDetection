import os
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

from torchvision.transforms import functional as F
# from transformers import BertTokenizer, CLIPProcessor, CLIPModel, CLIPTokenizer

from config.config_t5 import *
from dataset.dataset import *
from train_eval_t5 import train

from model.clip import *
from model.vit_roberta import *
from model.MHKE import *
from model.prompt_t5 import *
from model.prompthate import *

def get_model(config):
    if config.model_name == "clip":
        model = ChineseCLIPMemesClassifier(config).to(config.device)
        # model = MHKE_CLIP(config).to(config.device)
    elif config.model_name == "vit-roberta":
        model = VitRobertaMemesClassifier(config).to(config.device)
    elif config.model_name == "prompthate":
        model = PromptModel(config).to(config.device)
    elif config.model_name == "prompthate-t5":
        model = PromptT5Model(config).to(config.device)

    # 单模态
    elif config.model_name == "vit":
        model = VitClassifier(config).to(config.device)
    elif config.model_name == "resnet":
        model = ResNetClassifier(config).to(config.device)
    elif config.model_name == "roberta":
        model = RobertaClassifier(config).to(config.device)
    elif config.model_name == "bert":
        model = BertClassifier(config).to(config.device)
    elif config.model_name == "MHKE":
        model = MHKE(config).to(config.device)
    return model


if __name__ == '__main__':

    model_name = "prompthate-t5"
    task_name = "task_1"
    if model_name == "clip" or model_name == "vit-roberta":
        config = Config_fusion(model_name, task_name)
    elif model_name == "prompthate" or model_name == "prompthate-t5":
        config = Config_prompthate(model_name, task_name)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    parser = argparse.ArgumentParser(description='Hateful Meme Classification')
    parser.add_argument('--pad_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    # parser.add_argument('--start_ca_layer', type=int, default=24)
    parser.add_argument('--if_ear', type=str, default="True")
    parser.add_argument('--if_entity', type=str, default="True")
    parser.add_argument('--if_race', type=str, default="True")
    parser.add_argument('--if_att', type=str, default="False")
    parser.add_argument('--template', type=int, default=1)
    args = parser.parse_args()

    config.pad_size = args.pad_size
    # config.start_ca_layer = args.start_ca_layer
    config.learning_rate = args.lr
    config.if_ear = True if args.if_ear == "True" else False
    config.if_entity = True if args.if_entity == "True" else False
    config.if_race = True if args.if_race == "True" else False
    config.if_att = True if args.if_att == "True" else False
    config.template = args.template

    print(config.if_att)

    if model_name == "clip" or model_name == "vit-roberta":
        trn_data = MemeDataset(config, training=True)
        test_data = MemeDataset(config, training=False) 
    elif model_name == "prompthate" or model_name == "prompthate-roberta":
        trn_data = PromptMemeDataset(config, training=True)
        test_data = PromptMemeDataset(config, training=False) 
    elif model_name == "prompthate-t5":
        trn_data = PromptMemeT5Dataset(config, training=True)
        test_data = PromptMemeT5Dataset(config, training=False) 

    torch.save({
        'trn_data' : trn_data,
        'test_data' : test_data,
        }, config.data_path)

    print('The size of the Training dataset: {}'.format(len(trn_data)))
    print('The size of the Test dataset: {}'.format(len(test_data)))

    train_iter = DataLoader(trn_data, batch_size=int(config.batch_size), shuffle=False)
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)

    model = get_model(config)
    train(config, model, train_iter, test_iter)

    # for i in tqdm(range(len(trn_data))):
    #     if len(trn_data[i]["input_ids"]) > 256:
    #         print(trn_data[i]["input_ids"])
