import os
import csv
from inference import inference
import torch
import wandb


import numpy as np
import tensorflow as tf
from trainer import MambaTrainer
from transformers import TrainingArguments, OpenAIGPTConfig, OpenAIGPTLMHeadModel, GPTNeoConfig, GPTNeoForCausalLM
from tokenisation import ALPHABET_SIZE, detokenise_nt_seqs, additional_aa_token_to_index
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba.mamba_ssm.models.config_mamba import MambaConfig
from auxiliary_file import train_validate_test_split, compute_norm_levenshtein
from data import MambaDataModule

# Adapted from the master thesis of Jacob Marcel Anter "AI Method for Prediction of Protein Synthesis in Microorganisms".
def compute_metrics(eval_preds):
    """
    This function computes the accuracy and levenshtein norm for the predictions made by the model.
    
    """
    logits = eval_preds.predictions
    labels = eval_preds.label_ids

    predictions = logits.argmax(-1)
    
    for i, seq in enumerate(labels):
        for j, n in enumerate(seq):
            if (labels[i][j] == -100):
                labels[i][j] = 0

    match = labels == predictions

    mask = labels != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    levenshtein_list = []

    detokenized_preds = detokenise_nt_seqs(predictions)
    detokenized_labels  = detokenise_nt_seqs(labels)

    for i,pred in enumerate(detokenized_preds):
        current_levenshtein_distance = compute_norm_levenshtein(
            pred.split("<END>")[0].replace("<START>", ""), detokenized_labels[i].split("<END>")[0].replace("<START>", "")
        )
        levenshtein_list.append(current_levenshtein_distance)
 
    min_lv_dist = min(levenshtein_list)
    max_lv_dist = max(levenshtein_list)
    median_lv_dist = np.median(levenshtein_list)
    first_quartile_lv_dist = np.percentile(levenshtein_list, 25)
    third_quartile_lv_dist = np.percentile(levenshtein_list, 75)
    mean_lv_dist = np.mean(levenshtein_list)
    std_dev_lv_dist = np.std(levenshtein_list)

    return {"masked_accuracy": tf.reduce_sum(match)/tf.reduce_sum(mask), "minimum_levenshtein": min_lv_dist, "maximum_levenshtein": max_lv_dist, "median_levenshtein": median_lv_dist, "first_quartile_levenshtein": first_quartile_lv_dist, "third_quartile_levenshtein": third_quartile_lv_dist, "mean_levenshtein": mean_lv_dist, "standard_deviation": std_dev_lv_dist}




def create_no_duplicates_CSV(path_a, path_b, new_file_name):
    """
    If "out of sample" testing wants to be done, the genomes used for testing have to be put in an csv file at path_a and those for validation and training in a 
    csv file at path_b. This functions creates a file with the name new_file_name that contains the entries of the file at path_b that are not in the file at path_a.
    This way the new file can be used so that testing is not done on sequences on which the model was already trained.
    """
    already_seen_rows = []
    with open(path_a, 'r', newline='') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            already_seen_rows.append(row)

    with open(new_file_name, 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        with open(path_b, 'r', newline='') as input_file:
            reader = csv.reader(input_file)
            header = ["Amino acid sequence", "Nucleotide sequence"]
            writer.writerow(header)
            for row in reader:
                if row not in already_seen_rows:
                    writer.writerow(row)



dir = os.path.dirname(__file__)
path_a = os.path.join(dir, "Corynebacterium_glutamicum_genome_aa_nt_seqs.csv")
path_b = os.path.join(dir, "Corynebacterium_glutamicum_genomes_aa_nt_seqs.csv")
new_file_name = "Corynebacterium_glutamicum_genomes_no_duplicates_aa_nt_seqs.csv"


"""
create_no_duplicates_CSV(path_a, path_b, new_file_name)
_, validation_set, _ = train_validate_test_split(
    path_to_csv= path_a,
        train_size= 0.01,
        valid_size= 0.98
    )
training_set, _, _ = train_validate_test_split(
    path_to_csv= os.path.join(dir, new_file_name),
        train_size= 0.98,
        valid_size= 0.01
    )  
"""
training_set, test_set, validation_set = train_validate_test_split(
    path_to_csv= os.path.join(dir, "Corynebacterium_glutamicum_genomes_aa_nt_seqs.csv"),
        train_size=0.8,
        valid_size= 0.1
    ) 

data_module_train = MambaDataModule(
        data_set=training_set
    )
data_module_test = MambaDataModule(
        data_set=validation_set
    )



#config = MambaConfig(d_model=1280, n_layer=32, vocab_size=ALPHABET_SIZE) #-> 332 320 000
#config = MambaConfig(d_model=700, n_layer=17, vocab_size=ALPHABET_SIZE) #-> 53 446 400
#config = MambaConfig(d_model=480, n_layer=12, vocab_size=ALPHABET_SIZE) #-> 17 954 400 param
#config = MambaConfig(d_model=40, n_layer=1, vocab_size=ALPHABET_SIZE) #-> 17 440 param
#config = MambaConfig(d_model=120, n_layer=3, vocab_size=ALPHABET_SIZE) #-> 319 440
#config = MambaConfig(d_model=200, n_layer=5, vocab_size=ALPHABET_SIZE) #-> 1 377 600
#config = MambaConfig(d_model=16, n_layer=1, vocab_size=ALPHABET_SIZE) #-> 4544 param
#config = MambaConfig(d_model=400, n_layer=10, vocab_size=ALPHABET_SIZE) #-> 10 473 200 param
config = MambaConfig(d_model=240, n_layer=6, vocab_size=ALPHABET_SIZE) #-> 2 337 360 param
#config = MambaConfig(d_model=440, n_layer=11, vocab_size=ALPHABET_SIZE) #-> 13 889 040 param
#config = MambaConfig(d_model=320, n_layer=8, vocab_size=ALPHABET_SIZE) #-> 5 427 520 param
#config = MambaConfig(d_model=280, n_layer=7, vocab_size=ALPHABET_SIZE) #-> 3 671 920 param
#config = MambaConfig(d_model=80, n_layer=2, vocab_size=ALPHABET_SIZE) #-> 103 600 param
#config = MambaConfig(d_model=160, n_layer=4, vocab_size=ALPHABET_SIZE) #-> 722 720 param
model = MambaLMHeadModel(config, device="cuda", dtype=torch.float32)  
print(sum(p.numel() for p in model.parameters()))

# could try GPT here or GPT Neo
#configuration = OpenAIGPTConfig(vocab_size = ALPHABET_SIZE, n_positions= 1500, n_embd = 384)
#model = OpenAIGPTLMHeadModel(configuration)
#configuration = GPTNeoConfig(embed_dropout= 0.1, resid_dropout = 0.1, attention_dropout= 0.1, vocab_size = ALPHABET_SIZE, max_position_embeddings= 512, window_size = 50, hidden_size = 512, intermediate_size = 2048, num_layers = 6, attention_types = [[['global', 'local'], 3]],num_heads = 4, bos_token_id= additional_aa_token_to_index["<START>"], eos_token_id = additional_aa_token_to_index["<END>"])
#model = GPTNeoForCausalLM(configuration)


trainer = MambaTrainer(
    model=model,
    train_dataset=data_module_train.dataset,
    eval_dataset = data_module_test.dataset,
    args=TrainingArguments(
        num_train_epochs=20,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        output_dir="mymamba",
        logging_steps=50,
        # save_steps=500,
        per_device_eval_batch_size=1,
        save_strategy = "no",
        eval_strategy = "epoch",
        learning_rate=1e-4,
        weight_decay = 0.02,
        #adam_beta2 = 0.95,
        #adam_epsilon = 1e-9,
        warmup_ratio = 0.1,
        #max_grad_norm = 1
    ),
    compute_metrics=compute_metrics,
    data_collator=data_module_train.data_collator,
) 

trainer.train()
trainer.save_model("./trained_mamba-CURRENT")

