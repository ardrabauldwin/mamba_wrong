# Adapted from the master thesis of Jacob Marcel Anter "AI Method for Prediction of Protein Synthesis in Microorganisms".

import numpy as np
import torch 


# Amino acid tokenisation

# The enumeration of amino acids below does not contain selenocysteine
# (U) and the letter X, denoting any amino acid / an unknown amino acid
# This is due to the fact that the coding sequences were downloaded from
# NCBI, not the protein sequences, and nucleotide sequences containing
# ambiguous characters (such as N for any nucleotide) were excluded
# Hence, the amino acid character X does not occur
# Moreover, the default table from NCBI was used for translation, i. e.
# the amino acid selenocysteine does not occur in the protein sequences
# However, the asterisk is included, as it denotes a stop codon
ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY*'

ADDITIONAL_TOKENS = ['<PAD>', '<START>', '<END>']

# To each sequence, one <START> and <END> token is added
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
n_add_aa_tokens = len(ADDITIONAL_TOKENS)
aa_token_to_index = {
    aa: i + n_add_aa_tokens for i, aa in enumerate(ALL_AAS)
}
additional_aa_token_to_index = {
    token: i for i, token in enumerate(ADDITIONAL_TOKENS)
}
global_aa_token_to_index = {
    **additional_aa_token_to_index, **aa_token_to_index
}
index_to_aa_token = {
    index: token for token, index in global_aa_token_to_index.items()
}
N_AA_TOKENS = len(global_aa_token_to_index)


def _parse_seq(seq):
    """
    This function is responsible for converting an input sequence to a
    string, if applicable, and to return the input string otherwise.

    Parameters
    ----------
    seq: str or binary
        The input amino acid sequence to be tokenised.

    Returns
    -------
    seq: str
        The input amino acid sequence to be tokenised.
    """
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))


def _tokenise_aa_seq(seq):
    """
    This function performs tokenisation for amino acid sequences. This
    is achieved by splitting an amino acid sequence into individual
    tokens, i. e. individual amino acids, and replacing all individual 
    tokens with their unique integer index defined in the dictionary
    `token_to_index`.

    Apart from that, one <START> and <END> token is added to each
    sequence, respectively.

    Parameters
    ----------
    seq:  str or binary
        The amino acid sequence to be tokenised.

    Returns
    -------
    tokenised_seq: list of int
        The tokenised analogue of the input amino acid sequence.
    """
    tokenised_seq = (
        [additional_aa_token_to_index["<START>"]]
        +
        [
            aa_token_to_index.get(aa) for aa in _parse_seq(seq)
        ]
        +
        [additional_aa_token_to_index["<END>"]]
        
    )

    return tokenised_seq


def tokenise_aa_seqs(seqs):
    """
    This function performs tokenisation for a whole data set of amino
    acid sequences.

    Parameters
    ----------
    seqs: iterable, dtype=str or dtype=binary
        An iterable containing the amino acid sequences to be tokenised.

    Returns
    -------
    tokenised_seqs_array: list, dtype=torch.LongTensor
        A list harbouring the tokenised analogues of the input
        amino acid sequences, for each sequence a Tensor. 
    """
    tokenised_seqs_list = []

    for seq in seqs:
        tokenised_seqs_list.append(torch.LongTensor(_tokenise_aa_seq(seq)))
    
    return tokenised_seqs_list





# Nucleotide/codon tokenisation

# As it is exclusively dealt with DNA sequences, and not RNA sequences,
# uracil is omitted
# The letter N, denoting any nucleotide, is not included as sequences
# containing ambiguous nucleotides were discarded in advance in order to
# permit an unambiguous translation into amino acids
# Note that in the case of nucleotide sequences, tokenisation is not
# performed for the individual characters/nucleotides as with the amino
# acids, but rather for individual codons, i. e. triplets
ALL_CODONS = []
for nt_1 in "ATCG":
    for nt_2 in "ATCG":
        for nt_3 in "ATCG":
            ALL_CODONS.append(nt_1 + nt_2 + nt_3)

# As with the amino acid tokenisation, the '<OTHER>' token is omitted as
# an entirely unambiguous alphabet is used for the nucleotide
# tokenisation and sequences containing ambiguous characters have been
# discarded

ADDITIONAL_NT_TOKENS = ['<PAD>', '<START>', '<END>']

n_codons = len(ALL_CODONS)
n_additional_tokens = len(ADDITIONAL_NT_TOKENS)

codon_token_to_index = {
    codon: i + n_additional_tokens for i, codon in enumerate(ALL_CODONS)
}

additional_codon_token_to_index = {
    token: i for i, token in enumerate(ADDITIONAL_NT_TOKENS)
}
global_codon_token_to_index = {
    **additional_codon_token_to_index, **codon_token_to_index
}
index_to_codon_token = {
    index: token for token, index in global_codon_token_to_index.items()
}

N_CODON_TOKENS = len(global_codon_token_to_index) 

ALPHABET_SIZE = N_CODON_TOKENS

def _split_into_triplets(nt_seq):
    """
    This function splits an input nucleotide sequence into its
    individual codons / triplets.

    Parameters
    ----------
    nt_seq: str
        The nucleotide sequence to be split into triplets / codons.

    Returns
    -------
    codons: list, dtype=str
        The list containing the codons the input nucleotide sequence is
        made up of.
    """
    codons = []
    n_triplets = len(nt_seq) // 3

    for i in range(n_triplets):
        start_index = 3 * i
        end_index = 3 * (i + 1)
        triplet = nt_seq[start_index:end_index]
        codons.append(triplet)

    return codons


def _tokenise_nt_seq(seq):
    """
    This function is analogous to the function `_tokenise_aa_seq` and
    performs tokenisation for nucleotide sequences. However, instead of
    taking the individual nucleotide letters as tokens and mapping them
    to integer indices, three successive nucleotides, i. e. triplets /
    codons are taken as tokens. Hence, there are 64 unique tokens,
    without the additional tokens (e. g. <START>, <END>, etc.).

    Parameters
    ----------
    seq: str or binary
        The input nucleotide sequence to be tokenised.

    Returns
    -------
    tokenised_seq: list, dtype=int
        The tokenised analogue of the input nucleotide sequence.
    """
    tokenised_seq = (
        [additional_codon_token_to_index["<START>"]]
        +
        [
            codon_token_to_index.get(codon) for codon
            in _split_into_triplets(_parse_seq(seq))
        ]
        +
        [additional_codon_token_to_index["<END>"]]
    )

    return tokenised_seq


def tokenise_nt_seqs(seqs):
    """
    This function performs tokenisation for a whole data set of
    nucleotide sequences.

    Parameters
    ----------
    seqs: iterable, dtype=str or dtype=binary
        An iterable containing the nucleotide sequences to be tokenised.

    Returns
    -------
    tokenised_seqs_array: list, dtype = torch.LongTensor
        A list harbouring  the tokenised analogues of the input
        nucleotide sequences for each sequence a Tensor. 
    """
    tokenised_seqs_list = []

    for seq in seqs:
        tokenised_seqs_list.append(torch.LongTensor(_tokenise_nt_seq(seq)))
   
    return tokenised_seqs_list


def detokenise_nt_seqs(nt_token_array):
    """
    This function performs detokenisation for a whole data set of
    nucleotide sequences.

    Parameters
    ----------
    nt_token: ndarray, dtype= int, shape = (m, n)
        The tokens to be tokenised.

    Returns
    -------
    nt_seq_array: ndarray, dtype=str, shape= (m, )
        The detokenised nucleotid sequences.
    """
    nt_seq_list = []
    
    for token_seq in nt_token_array:
        codon_list = []
        for token in token_seq:
            str = "-"
            if not index_to_codon_token.get(token) is None:
                str = index_to_codon_token.get(token)
            codon_list.append(str)
        codon_str = "".join(codon_list)
        nt_seq_list.append(codon_str)
    
    return nt_seq_list


def format_dataset_for_predictive_alignment(aa_nt_pairs):
    """
    This function formats the data set into a shape digestible by the
    transformer/mamba architecture. To be more precise, a tuple `(inputs,
    targets)` is returned.

    Parameters
    ----------
    aa_nt_pairs: Pandas DataFrame, shape=(n, 2)
        The Pandas DataFrame containing the data set. It consists of two columns / series, of
        which the first represents the amino acid sequences and the
        second one represents the corresponding nucleotide sequences.

    Returns
    -------
    output_tuple: tuple, shape=(2, )
        Tuple containing the input and target. The first
        element is the tokenised amino acid sequence, the second the tokenised
        nucleotide sequence.
    """
    # As a first step, convert the Pandas DataFrame into a NumPy array
    # as the latter is easier to handle with respect to iteration, etc.
    aa_nt_pairs = aa_nt_pairs.to_numpy()

    aa_seqs, nt_seqs = zip(*aa_nt_pairs)
    # The zip function generates an iterator of tuples; convert them to
    # lists
    aa_seqs = list(aa_seqs)
    nt_seqs = list(nt_seqs)


    aa_encoding = tokenise_aa_seqs(aa_seqs)
    nt_encoding = tokenise_nt_seqs(nt_seqs)

    num_data_points = 0
    for l in aa_encoding:
        num_data_points += l.size(dim=0)

    print("The number of data points: " + str(num_data_points))
    return (aa_encoding, nt_encoding)