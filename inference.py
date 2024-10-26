from data import DataCollatorForMambaDataset
from tokenisation import detokenise_nt_seqs, tokenise_aa_seqs
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch
import numpy as np

def inference(aa_seqs, model_name):
    """
    This function does inference on the amino acid sequences.

    Parameters
    ----------
    aa_seqs: list, dtype = str
        The amino acid sequences that shall be translated by the model.
    model_name: string
        The name of the model to use.

    Returns
    -------
    split_results: list
        A list of lists, each list containing the strings for each nucleotid triplet.
    """
    tokenised_seqs_list = tokenise_aa_seqs(aa_seqs)
    collator = DataCollatorForMambaDataset()
    sequence_of_dicts = []
    for seq in tokenised_seqs_list:
        sequence_of_dicts.append(dict(input_ids=seq, labels=torch.zeros(1)))
    tokenised_seqs_list = collator(sequence_of_dicts)["input_ids"]
    
    input_seqs = torch.from_numpy(np.array(tokenised_seqs_list)).to("cuda")
    model = MambaLMHeadModel.from_pretrained(model_name, device="cuda", dtype=torch.float32)
    results = []
    split_results = []
    for seq in input_seqs:
        output = model(input_seqs)
        lm_logits = output.logits
        predictions = lm_logits.argmax(-1)
        detokenized = detokenise_nt_seqs(predictions.cpu().numpy())[0]
        results.append(detokenized.split("<END>")[0].replace("<START>", ""))   
    
    for result in results:
        list = [result[i:i+3] for i in range(0, len(result), 3)]
        split_results.append(list)

    return split_results

inputseq = ["MITLTNVRKEYSSDAIGPVNL", "MITLT"]
inference(inputseq, "./trained_mamba-CURRENT")


