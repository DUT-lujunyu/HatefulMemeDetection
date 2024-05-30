import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, ViTModel
from model.modeling_flan_t5 import T5ForMultimodalGeneration


class PromptT5Model(nn.Module):
    def __init__(self, config):
        super(PromptT5Model, self).__init__()
        self.config = config
        self.device = config.device 
        self.model_path = config.flan_t5_base_path
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)

        # Multi-modal
        if self.config.if_visual:
            self.engine = T5ForMultimodalGeneration.from_pretrained(self.model_path, config.patch_size)
            self.extractor = ViTModel.from_pretrained(config.vit_path)
            for name, param in self.extractor.named_parameters():
                param.requires_grad = False
        else:
            self.engine = T5ForConditionalGeneration.from_pretrained(self.model_path)

    def forward(self, **args):
        input_ids = args["prompt_input_ids"].to(self.device)
        input_masks = args["prompt_attention_mask"].to(self.device)
        output_ids = args["output_ids"].to(self.device)
        output_masks = args["output_mask"].to(self.device)
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100      
        
        # Multi-modal
        if self.config.if_visual:
            image_ids = self.extractor(pixel_values = args['image_tensor'].to(self.device))['last_hidden_state']
            output = self.engine(input_ids, image_ids, attention_mask=input_masks, decoder_input_ids=None,
                        decoder_attention_mask=output_masks, labels=output_ids, output_attentions=True)
        else:
            output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                        decoder_attention_mask=output_masks, labels=output_ids, output_attentions=True)

        # loss = output[0]
        return output

    def generate(self, **args):
        input_ids = args["prompt_input_ids"].to(self.device)
        input_masks = args["prompt_attention_mask"].to(self.device)
        if self.config.if_visual:
            image_ids = self.extractor(pixel_values = args['image_tensor'].to(self.device))['last_hidden_state']
            output = self.engine.generate(input_ids=input_ids, image_ids=image_ids, attention_mask=input_masks, 
                                        max_length=self.config.label_pad_size)
        else:
            output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                        max_length=self.config.label_pad_size)
        
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **args):
        input_ids = args["prompt_input_ids"].to(self.device)
        input_masks = args["prompt_attention_mask"].to(self.device)
        if self.config.if_visual:
            image_ids = self.extractor(pixel_values = args['image_tensor'].to(self.device))['last_hidden_state']
            output = self.engine.generate(input_ids=input_ids, image_ids=image_ids, attention_mask=input_masks, 
                                        max_length=self.config.label_pad_size)
        else:
            output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                        max_length=self.config.label_pad_size)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output
    