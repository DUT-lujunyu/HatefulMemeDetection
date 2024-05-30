import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from os import path

class Config_base(object):

    """配置参数"""
    def __init__(self, model_name, task_name):

        # path
        self.model_name = model_name
        self.task_name = task_name

        self.model_base_path = "/home/home_ex/lujunyu/models/"
        self.chinese_clip_path = self.model_base_path + "chinese-clip-vit-base-patch16"
        self.clip_path = self.model_base_path + "clip-vit-base-patch32"
        
        self.chinese_roberta_path = self.model_base_path + "chinese-roberta-wwm-ext"
        self.chinese_bert_path = self.model_base_path + "bert-base-chinese"
        self.roberta_base_path = self.model_base_path + "roberta-base"
        self.roberta_large_path = self.model_base_path + "roberta-large"
        self.flan_t5_base_path = self.model_base_path + "flan-t5-base"

        self.vit_path = self.model_base_path + "vit-base-patch16-224" 
        self.resnet_path = self.model_base_path + "resnet-50" 

        # self.meme_path = path.dirname(path.dirname(__file__)) + '/meme/'          
        # self.train_path = path.dirname(path.dirname(__file__)) + '/train_data_discription.json'                                # 训练集
        # self.dev_path = path.dirname(path.dirname(__file__)) + '/test_data_discription.json'                                    # 验证集
        # self.test_path = path.dirname(path.dirname(__file__))+'/test_data_discription.json'  

        self.meme_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/img/'          
        self.train_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/train.csv'                                # 训练集
        self.dev_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/test.csv'   
        self.test_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/test.csv'
        # self.train_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/mem_train_capattr_clean.json'                                # 训练集
        # self.dev_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/mem_test_capattr_clean.json'   
        # self.test_path = path.dirname(path.dirname(__file__)) + '/resource/hateful_memes/mem_test_capattr_clean.json'

        self.result_path = path.dirname(path.dirname(__file__))+'/result'   
        self.prediction_path = path.dirname(path.dirname(__file__))+'/prediction'                              # 测试集
        self.checkpoint_path = path.dirname(path.dirname(__file__))+'/saved_dict'        # 数据集、模型训练结果
        self.data_path = self.checkpoint_path + '/' + self.model_name + '_data.tar'

        # language
        # self.language = "CN"

        # model
        self.frozen_layers = 0  # 被冻结的层     
        self.dropout = 0.5                                              # 随机失活
        self.fc_hidden_dim = 256
        self.weight = 0.5        
        self.learning_rate = 1e-5                                       # 学习率  transformer:5e-4 
        self.num_epochs = 30                                            # epoch数 
        self.num_warm = 0                                              # 预热
        self.batch_size = 16                                           # mini-batch大小
        self.weight_decay = 0.01
        self.eps = 1e-8

        if self.task_name == "task_1":
            self.num_classes = 2                                             # 类别数
        else:
            self.num_classes = 5                                             # 类别数

        # dataset
        self.seed = 1        
        self.pad_size = 256                                              # 每句话处理成的长度(短填长切)
        self.add_entity = False
        self.add_race = False
        self.add_attribute = False

        # train
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   # 设备

        # evaluate
        self.score_key = "F1"                                            # 评价指标


# text and image feature fusion (concat/attention)  
class Config_fusion(Config_base):

    """配置参数"""
    def __init__(self, model_name, task_name):
        super().__init__(model_name, task_name)

        if self.model_name == "clip":  
            self.text_encoder = "clip"
            self.image_encoder = "clip"             
            self.hidden_dim = 512
        else:
            self.image_encoder = "vit"   
            self.text_encoder = "chinese-roberta-base"
            self.hidden_dim = 768 


class Config_prompthate(Config_base):

    def __init__(self, model_name, task_name):
        super().__init__(model_name, task_name) 

        if model_name == "prompthate":
            self.text_encoder = "roberta-large"
            self.hidden_dim = 1024
            self.label_list = ["good", "bad"]         
        elif model_name == "prompthate-t5":
            self.text_encoder = "flan-t5-base"
            self.hidden_dim = 768
            self.label_list =  {0: "good", 1: "bad"}     

        self.image_encoder = "vit"  
        self.patch_size = 768
        self.label_pad_size = 3  # prompthate_t5 的 output 编码最大长度

        self.if_visual = True  # 是否引入图像信息
        self.start_ca_layer = 24  # large：<= 24, base: <= 12

        self.if_ear = False  # 是否引入信息熵损失函数
        self.ear_reg_strength = 0.01
        self.template = 1  # 使用的 template 1
