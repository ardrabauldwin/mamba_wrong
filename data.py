# Adapted from https://github.com/havenhq/mamba-chat
 
import torch
from dataclasses import dataclass
from typing import Dict, Sequence
from torch.utils.data import Dataset
from tokenisation import additional_aa_token_to_index, additional_codon_token_to_index, format_dataset_for_predictive_alignment

class MambaDataset(Dataset):
    """
    Stores the tokenised inputs and labels.
    """
    def __init__(self, data_set):
        super(MambaDataset, self).__init__()
        
        data_dict = preprocess(data_set)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForMambaDataset(object):
    """
    Pads batch of labels and inputs
    """
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=additional_aa_token_to_index['<PAD>'])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=additional_codon_token_to_index['<PAD>'])

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(additional_aa_token_to_index['<PAD>']),
        )
    

class MambaDataModule():
    def __init__(self, data_set):
        self.dataset = MambaDataset(data_set)
        self.data_collator = DataCollatorForMambaDataset()
        

def preprocess(dataset) -> Dict:
    all_input_ids, all_label_ids = format_dataset_for_predictive_alignment(dataset)

    return dict(input_ids=all_input_ids, labels=all_label_ids)