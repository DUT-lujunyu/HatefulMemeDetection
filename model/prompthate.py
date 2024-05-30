import torch
import torch.nn as nn
# from transformers import RobertaForMaskedLM
from transformers import BertForMaskedLM, BertTokenizer, RobertaTokenizer, ViTModel
from model.modeling_roberta import RobertaForMaskedLM


class PromptModel(nn.Module):
    def __init__(self, config):
        super(PromptModel, self).__init__()
        self.config = config
        self.label_word_list = config.label_list 
        # self.model_path = config.chinese_roberta_path
        # self.bert = BertForMaskedLM.from_pretrained(self.model_path)
        self.model_path = config.roberta_large_path
        self.tokenzier = RobertaTokenizer.from_pretrained(self.model_path)

        if config.if_visual:
            self.roberta = RobertaForMaskedLM.from_pretrained(self.model_path, patch_size=config.patch_size, start_ca_layer=config.start_ca_layer)
            self.extractor = ViTModel.from_pretrained(config.vit_path)
            for name, param in self.extractor.named_parameters():
                param.requires_grad = False
        else:
            self.roberta = RobertaForMaskedLM.from_pretrained(self.model_path)

        self.device = config.device

    def forward(self, **args):
        label_ids_list = self.tokenzier.encode(self.label_word_list, add_special_tokens=False)
        input_ids = args["prompt_input_ids"].to(self.device)
        attention_mask = args["prompt_attention_mask"].to(self.device)
        mask_pos = args["mask_pos"].to(self.device)
        batch_size = input_ids.size(0)

        # the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        if self.config.if_visual: 
            image_ids = self.extractor(pixel_values = args['image_tensor'].to(self.device))['last_hidden_state']
            out = self.roberta(input_ids, attention_mask, image_ids=image_ids, output_attentions=True)             
        else:  
            out = self.roberta(input_ids, attention_mask, output_attentions=True)

        prediction_mask_scores = out[0][torch.arange(batch_size), mask_pos]
        logits = []
        for label_id in range(len(label_ids_list)):
            logits.append(prediction_mask_scores[:, label_ids_list[label_id]].unsqueeze(-1))
            
        logits = torch.cat(logits, -1)
        return logits, out  # [batch_size, num_labels]
        

# # bert, chinese-bert, chinese-roberta    
class BertPromptModel(nn.Module):
    def __init__(self, config):
        super(BertPromptModel, self).__init__()
        self.label_word_list = config.label_list 
        self.path = config.chinese_roberta_path
        self.roberta = BertForMaskedLM.from_pretrained(self.path)
        self.device = config.device

    def forward(self, **args):
        tokens = args["input_ids"].to(self.device)
        attention_mask = args["attention_mask"].to(self.device)
        mask_pos = args["mask_pos"].to(self.device)
        batch_size = tokens.size(0)

        # the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            
        out = self.roberta(tokens, attention_mask)
        prediction_mask_scores = out[0][torch.arange(batch_size), mask_pos]
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            # print(prediction_mask_scores[:, self.label_word_list[label_id]].shape)
            
        logits = torch.cat(logits, -1)
        return logits  # [batch_size, num_labels]

    