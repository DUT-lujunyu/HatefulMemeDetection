import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification
from collections import namedtuple


def compute_negative_entropy(
    inputs: tuple, attention_mask: torch.torch, return_values: bool = False
):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS
    """
    inputs = torch.stack(inputs)  #  LayersBatchHeadsSeqlenSeqlen
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    #  average over attention heads
    pool_heads = inputs.mean(2)  # Layers Batch Heads/2 Seqlen Seqlen

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]
        # print(sample.shape)
        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy


EARClassificationOutput = namedtuple(
    "EARClassificationOutput",
    ["model_output", "negative_entropy", "reg_loss", "loss"]
)


class EARModelForSequenceClassification(torch.nn.Module):

    def __init__(self, model_name_or_path, ear_reg_strength: float = 0.01):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.ear_reg_strength = ear_reg_strength

    def forward(self, **model_kwargs):
        output = self.model(**model_kwargs, output_attentions=True)
        # print(len(output))
        # print(output["logits"].shape)
        # print(output["attentions"][0].shape)
        '''
        output:
            loss
            logits [batch_size, 2]
            hidden_states 
            attentions turple类型 包含12个元素（12层），每层 [batch_size, heads, Seq_len, Seqlen]
        '''
        negative_entropy = compute_negative_entropy(
            output["attentions"], model_kwargs["attention_mask"]
        )
        reg_loss = self.ear_reg_strength * negative_entropy
        if len(output) == 2:
            loss = reg_loss
        else:
            loss = reg_loss + output["logits"]

        return EARClassificationOutput(
            model_output=output,
            negative_entropy=negative_entropy,
            reg_loss=reg_loss,
            loss=loss
        )