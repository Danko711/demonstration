from typing import List, Mapping, Union
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self,
                 inputs: Mapping[str, torch.Tensor],
                 labels: List[Union[str, int]] = None,
                 bert_inputs: List[str] = None
                 ):
        self.inputs = inputs
        self.labels = labels
        if bert_inputs is None:
            self.bert_inputs = ['features', 'attention_mask']
        else:
            self.bert_inputs = bert_inputs

    def __getitem__(self, idx):
        output_dict = {}
        for bert_inp in self.bert_inputs:
            output_dict[bert_inp] = self.inputs[bert_inp][idx]
        if self.labels is not None:  # targets
            output_dict['targets'] = self.labels[idx]
        return output_dict

    def __len__(self):
        return len(self.inputs[self.bert_inputs[0]])
