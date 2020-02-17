import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_bert import BertPooler


class CustomBertPooler(BertPooler):
    def __init__(self, config, pool_ind, num_hidden):
        super(CustomBertPooler, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size * num_hidden, config.hidden_size * num_hidden)
        self.pool_ind = pool_ind
        self.num_hidden = num_hidden

    def forward(self, hidden_states):
        # "pool" the model by simply taking FOUR last hidden states corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, self.pool_ind]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertClassifierClsTokens(nn.Module):
    """
    Simplified version of the same class by HuggingFace.
    See transformers/modeling_bert.py in the transformers repository.
    """

    def __init__(self, pretrained_model_name: str, num_classes: int = None, num_hidden: int = None,
                 num_additional_special_tokens: int = None):
        """
        Args:
            pretrained_model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_classes (int): the number of class labels
                in the classification task
        """
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes)
        config.output_hidden_states = True
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        if num_additional_special_tokens is not None:
            self.bert.resize_token_embeddings(30522 + num_additional_special_tokens)
        self.pooler_question = CustomBertPooler(config, 0, num_hidden)
        self.pooler_answer = CustomBertPooler(config, 1, num_hidden)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_question = nn.Linear(config.hidden_size * num_hidden, 21)
        self.classifier_answer = nn.Linear(config.hidden_size * num_hidden, 9)

    def forward(self, features, attention_mask=None, token_type_ids=None):
        """Compute class probabilities for the input sequence.

        Args:
            features (torch.Tensor): ids of each token,
                size ([bs, seq_length]
            attention_mask (torch.Tensor): binary tensor, used to select
                tokens which are used to compute attention scores
                in the self-attention heads, size [bs, seq_length]
            token_type_ids (torch.Tensor): Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in [0, 1]: 0 corresponds to a "sentence A" token, 1
            corresponds to a "sentence B" token
        Returns:
            PyTorch Tensor with predicted class probabilities
        """
        assert attention_mask is not None, "attention mask is none"
        outputs = self.bert(features,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        if self.num_hidden == 1:
            last_hidden = outputs[2:][0][-1]
        else:
            last_hidden = torch.cat(outputs[2:][0][-self.num_hidden:], dim=2)
        pooled_output_question = self.pooler_question(last_hidden)
        pooled_output_question = self.dropout(pooled_output_question)

        pooled_output_answer = self.pooler_answer(last_hidden)
        pooled_output_answer = self.dropout(pooled_output_answer)

        logits_question = self.classifier_question(pooled_output_question)
        logits_answer = self.classifier_answer(pooled_output_answer)

        return torch.cat([logits_question, logits_answer], dim=1)
