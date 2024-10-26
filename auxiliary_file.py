# Adapted from the master thesis of Jacob Marcel Anter "AI Method for Prediction of Protein Synthesis in Microorganisms".

import os
import csv
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import Levenshtein

def create_csv_file(path_to_genomes, host_name, max_genome_num=None, max_seq_len=None):
    """
    This function generates a CSV file comprising all the genomes used
    for training, validating and testing.
    
    The whole data set is shuffled and subsequently
    subjected to the train-validate-test split.

    The CSV file contains two columns, of which the first represents
    amino acid sequences of coding genes and the second represents the
    respective nucleotide sequence.

    Additionally, a second file is generated, which is a text file
    containing information regarding the sequences in the CSV file, such
    as the total amount of sequences meeting the prerequisites (only
    unambiguous alphabet, sequence length is a multiple of 3).

    A third file is generated, too. This third file is a text file
    listing all the genomes incorporated in the respective CSV file.

    Parameters
    ----------
    path_to_genomes: str
        A string denoting the path to the directory containing the
        genomes to be processed.
    host_name: str
        The name of the expression host for which the CSV file is
        supposed to be created.
    max_genome_num: int, optional
        The maximum amount of genomes to be incorporated in the CSV
        file. If this optional parameter is not specified in the
        function call, all genomes in the specified directory are
        incorporated.
    max_seq_len: int, optional
        The maximum length sequences to be incorporated are allowed to
        have (optional). 
    """
    genome_files_list = os.listdir(path_to_genomes)

    total_amount_of_genomes = len(genome_files_list)

    if max_genome_num is None:
        max_genome_num = total_amount_of_genomes

    incorporated_genomes_list = []

    CDS_counter = 0
    genome_counter = 0
    max_nt_seq_len = 0

    # The "newline" parameter below must be specified as an empty string
    # in order to prevent subsequent lines from being separated by an
    # empty line
    with open(f"{host_name}_aa_nt_seqs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Amino acid sequence", "Nucleotide sequence"]
        writer.writerow(header)
        rows = []
        # Iterate through the entire directory, translate each
        # nucleotide sequence into the respective amino acid sequence
        # and add the row to the CSV file
        for _, filename in enumerate(genome_files_list):
            if genome_counter == max_genome_num or not filename.endswith("fna"):
                continue
            
            genome_counter += 1
            CDS_this_genome = 0
            

            file_path = os.path.join(path_to_genomes, filename)
            for _, nt_seq in fasta.FastaFile.read_iter(file_path):
                # A translation of the complete nucleotide sequence
                # requires the sequence length to be a multiple of 3,
                # which is why sequences not meeting this requirement
                # are discarded
                current_nt_seq_len = len(nt_seq)
                if (current_nt_seq_len % 3) != 0:
                    continue
                # Skip sequences whose length exceeds the user-defined
                # maximum
                # As one token is later generated for three nucleotides
                # / one codon, the amount of amino acid residues and the
                # amount of tokens are the same
                if max_seq_len is not None:
                    current_aa_seq_len = current_nt_seq_len // 3
                    if current_aa_seq_len > max_seq_len:
                        continue
                # Furthermore, in order for the translation process to
                # successfully take place, the sequences may exclusively
                # contain the letters of the unambiguous alphabet, i. e.
                # the letters A, T, G and C
                # All sequences containing letters of the ambiguous
                # alphabet are therefore discarded
                if "R" in nt_seq:
                    continue
                if "Y" in nt_seq:
                    continue
                if "W" in nt_seq:
                    continue
                if "S" in nt_seq:
                    continue
                if "M" in nt_seq:
                    continue
                if "K" in nt_seq:
                    continue
                if "H" in nt_seq:
                    continue
                if "B" in nt_seq:
                    continue
                if "V" in nt_seq:
                    continue
                if "D" in nt_seq:
                    continue
                if "N" in nt_seq:
                    continue

                

                # Evaluate whether the sequence currently dealt with
                # has the largest length so far
                if current_nt_seq_len > max_nt_seq_len:
                    max_nt_seq_len = current_nt_seq_len

                # Presumably due to sequencing inaccuracies, the first
                # nucleotide sometimes is different from A, although the
                # second and third ones are T and G, respectively
                # Hence, in order to restore the start codon ATG, the
                # single nucleotide exchange is performed
                if nt_seq[0] != "A":
                    nt_seq = "A" + nt_seq[1:]
                
                # Now, perform the translation into the amino acid
                # sequence and add the entry to the CSV file
                aa_seq = seq.NucleotideSequence(nt_seq).translate(
                    complete=True
                )
                if [aa_seq, nt_seq] not in rows: 
                    # The sequence currently dealt with meets the
                    # prerequisites for being processed and is not a duplicate, which is why the
                    # counter for coding sequences is incremented by 1
                    CDS_counter += 1
                    CDS_this_genome += 1
                    rows.append([aa_seq, nt_seq])
            
            incorporated_genomes_list.append(filename + ": " + str(CDS_this_genome) + " sequences"+"\n")

        writer.writerows(rows)

    # The text file containing information regarding the sequences in the CSV file is generated
    # Logically, the maximum amino acid sequence length equals the
    # maximum nucleotide sequence length divided by three, as one codon
    # comprising of three nucleotides encodes one amino acid
    max_aa_seq_len = int(max_nt_seq_len / 3)

    lines = [
        f"The maximum sequence length was set to {max_seq_len}.\n"
        "The total amount of coding sequences meeting the prerequisites"
        f" is {CDS_counter}.\n",
        f"The largest nucleotide sequence encompasses {max_nt_seq_len} "
        "nt.\n",
        "Accordingly, the largest amino acid sequence encompasses "
        f"{max_aa_seq_len} amino acids.\n",
        "Keep in mind that the number of tokens, however, is the same, "
        "irrespective of whether nucleotides or amino acids are dealt\n"
        "with, as one codon and one amino acid are considered as one "
        "token, respectively!"
    ]

    with open(f"Information_on_{host_name}_csv_file.txt", "w") as f:
        f.writelines(lines)

    with open(
        f"Incorporated_sequences_in_{host_name}_csv_file.txt", "w"
    ) as f:
        f.write(
            f"The following {genome_counter} genomes were incorporated in "
            "the CSV file:\n"
        )
        f.writelines(incorporated_genomes_list)

def compute_norm_levenshtein(str_1, str_2):
    """
    This function computes the normalised Levenshtein distance between
    two strings, which is beneficial for the comparison of sequences
    that are of unequal length.

    Parameters
    ----------
    str_1: string
        The first of the two strings to perform the distance computation
        on.
    str_2: string
        The second of the two strings to perform the distance
        computation on.

    Returns
    -------
    normalised_levenshtein: float
        The normalised Levenshtein distance
    """
    unnormalised_levenshtein = Levenshtein.distance(str_1, str_2)
    max_length = max(len(str_1), len(str_2))
    normalised_levenshtein = (
        (max_length - unnormalised_levenshtein) / max_length
    )

    return normalised_levenshtein

def train_validate_test_split(path_to_csv, train_size, valid_size):
    """
    This function reads in a CSV file and performs a train-validate-test
    split on the data by applying scikit-learn's train_test_split twice
    in succession.
    """
    aa_nt_df = pd.read_csv(path_to_csv)

    training_set, remaining_set = train_test_split(
        aa_nt_df,
        train_size=train_size,
        random_state=0
    )

    # It must be kept in mind that the basic value the validation
    # set size is referring to now is different from the whole CSV file
    # Hence, this relative quantity must be adjusted
    valid_size = valid_size / (1 - train_size)
    validation_set, test_set = train_test_split(
        remaining_set,
        train_size=valid_size,
        random_state=0
    )

    return training_set, validation_set, test_set



data_dir = os.path.join(os.path.dirname(__file__), "data")
path_to_E_coli_genomes = os.path.join(data_dir, "escherichia")
path_to_B_subtilis_genomes = os.path.join(data_dir, "bacillus")
path_to_Corynebacterium_glutamicum_genomes = os.path.join(data_dir, "corynebacterium")


if __name__ == "__main__":
    create_csv_file(
        path_to_genomes=path_to_Corynebacterium_glutamicum_genomes,
        host_name="Corynebacterium_glutamicum_genomes",
        max_genome_num=58,
        max_seq_len=100000
    ) 
    """ create_csv_file(
        path_to_genomes=path_to_E_coli_genomes,
        host_name="E_coli_genomes",
        max_genome_num=6,
        max_seq_len=100000
    ) 
    create_csv_file(
        path_to_genomes=path_to_B_subtilis_genomes,
        host_name="B_subtilis_genomes",
        max_genome_num=6,
        max_seq_len=100000
    )    """
    

    