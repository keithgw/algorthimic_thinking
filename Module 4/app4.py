"""
Provide code and solution for Application 4
"""

import math
import random
import urllib2
import matplotlib.pyplot as plt
import project4 as seq
import re
import numpy as np

# URLs for data files
PAM50_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_PAM50.txt"
HUMAN_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_HumanEyelessProtein.txt"
FRUITFLY_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_FruitflyEyelessProtein.txt"
CONSENSUS_PAX_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_ConsensusPAXDomain.txt"
WORD_LIST_URL = "http://storage.googleapis.com/codeskulptor-assets/assets_scrabble_words3.txt"



###############################################
# provided code

def read_scoring_matrix(filename):
    """
    Read a scoring matrix from the file named filename.  

    Argument:
    filename -- name of file containing a scoring matrix

    Returns:
    A dictionary of dictionaries mapping X and Y characters to scores
    """
    scoring_dict = {}
    scoring_file = urllib2.urlopen(filename)
    ykeys = scoring_file.readline()
    ykeychars = ykeys.split()
    for line in scoring_file.readlines():
        vals = line.split()
        xkey = vals.pop(0)
        scoring_dict[xkey] = {}
        for ykey, val in zip(ykeychars, vals):
            scoring_dict[xkey][ykey] = int(val)
    return scoring_dict



def read_protein(filename):
    """
    Read a protein sequence from the file named filename.

    Arguments:
    filename -- name of file containing a protein sequence

    Returns:
    A string representing the protein
    """
    protein_file = urllib2.urlopen(filename)
    protein_seq = protein_file.read()
    protein_seq = protein_seq.rstrip()
    return protein_seq


def read_words(filename):
    """
    Load word list from the file named filename.

    Returns a list of strings.
    """
    # load assets
    word_file = urllib2.urlopen(filename)
    
    # read in files as string
    words = word_file.read()
    
    # template lines and solution lines list of line string
    word_list = words.split('\n')
    print "Loaded a dictionary with", len(word_list), "words"
    return word_list


def align_eyeless(scoring_matrix):
    """
    compute the local alignment and score of the human eyeless AA sequence and
    the drosophila eyeless AA sequence, using the PAM 50 scoring matrix
    """
    # load eyeless AA strings 
    human = read_protein(HUMAN_EYELESS_URL)
    drosophila = read_protein(FRUITFLY_EYELESS_URL)
    
    # compute local alignment matrix
    la_mtrx = seq.compute_alignment_matrix(human, drosophila, scoring_matrix, False)
    
    # compute local alignment
    return seq.compute_local_alignment(human, drosophila, scoring_matrix, la_mtrx)
    

def pax_domain(scoring_matrix, local_alignment):
    """
    Compare the local alignments of human and drosophila eyeless proteins to
    the consesus PAX domain by computing a global alignment.
    Return a tuple of percentages: one for human vs consensus, one for 
    drosophila vs consesus, each of which reports how many AAs are the same.
    """
    
    # load consesus pax domain
    pax = read_protein(CONSENSUS_PAX_URL)
    
    # remove dashes from local alignemnts (human and drosophila)
    human = re.sub('-', '', local_alignment[1])
    drosophila = re.sub('-', '', local_alignment[2])
    
    # compute global alignment for dash-less local alignments vs consesus
    human_pax_matrix = seq.compute_alignment_matrix(human, pax, scoring_matrix)
    human_pax = seq.compute_global_alignment(human, pax, scoring_matrix, human_pax_matrix)
    
    drosophila_pax_matrix = seq.compute_alignment_matrix(drosophila, pax, scoring_matrix)
    drosophila_pax = seq.compute_global_alignment(drosophila, pax, scoring_matrix, drosophila_pax_matrix)
        
    # compute counts of elements that agree in the two global alignments
    n_human_pax = len(human_pax[1])
    count_human_pax = 0.0
    for aa in range(n_human_pax):
        if human_pax[1][aa] == human_pax[2][aa]:
            count_human_pax += 1
    
    n_drosophila_pax = len(drosophila_pax[1])
    count_drosophila_pax = 0.0
    for aa in range(n_drosophila_pax):
        if drosophila_pax[1][aa] == drosophila_pax[2][aa]:
            count_drosophila_pax +=1
    
    # return proportion of agreement for two global alignments    
    return (count_human_pax / n_human_pax, count_drosophila_pax / n_drosophila_pax)
    
def generate_null_distribution(seq_x, seq_y, scoring_matrix, num_trials):
    """
    Inputs:
        seq_x, seq_y: character strings that share a common alphabet with 
            scoring_matrix.
        scoring_matrix: output of build_scoring_matrix. Dictionary of 
            dictionaries whose [seq_x[i]][seq_y[j]] value is the score of the
            alignment of seq_x[i], seq_y[i].
        num_trials: integer number of simulations to run
    Output:
        scoring_distribution: a list of scores from the simulations.
        
    Randomly shuffle seq_y num_trial times, score the local alignment with 
    seq_x.
    """
    # initialize
    scores = []
    
    # run trials
    for trial in range(num_trials):
        # shuffle seq_y
        _seq_y = list(seq_y)
        random.shuffle(_seq_y)
        rand_y = ''.join(_seq_y)
        
        # compute local alignment of seq_x and random permutation of seq_y
        alignment = seq.compute_alignment_matrix(seq_x, rand_y, scoring_matrix, False)
        score = seq.compute_local_alignment(seq_x, rand_y, scoring_matrix, alignment)[0]
        
        # update frequency distribution
        scores.append(score)
            
    return scores
            
def plot_htest(scores):
    """
    Input: A list of scores
    Output: A histogram of normalized scores
    Plot the normalized frequency distribution of scores of randomly generated
    amino acid sequence alignments.
    """
    plt.figure()
    plt.hist(scores, normed=True, color='c')
    plt.title('Histogram of Scores of Aligned, Randomly Permuted AA Sequences')
    plt.xlabel('Score of Local Alignment')
    plt.ylabel('Normalized Frequency')
    plt.show()
    
def zscore(scores, h_a):
    """
    Inputs:
            scores: list of scores generated under the null hypothesis
            h_a: score of local alignment of human and drosophila eyeless
    Compute the mean, standard deviation of the null distribution.
    Compute the z-score of the score of the local alignment of human and
    drosophila eyeless under the null hypothesis.
    """
    mu = np.mean(scores)
    sigma = np.std(scores)
    zscore = (h_a - mu) / sigma
    return (mu, sigma, zscore)
    
            
def run():
    
    # load sequences and scoring matrix
    human = read_protein(HUMAN_EYELESS_URL)
    drosophila = read_protein(FRUITFLY_EYELESS_URL)
    pam50 = read_scoring_matrix(PAM50_URL)
    
    # Question 1
    local_alignment = align_eyeless(pam50)
    print "Question 1 \n"
    print local_alignment
    print '=' * 30
    
    # Question 2
    pax_alignment = pax_domain(pam50, local_alignment)
    print "Question 2 \n"
    print 'Human/Pax Percent:', pax_alignment[0] * 100, '%'
    print '=' * 30
    
    # Question 4
    random.seed(6022)
    NSIM = 1000
    h_0 = generate_null_distribution(human, drosophila, pam50, NSIM)
    plot_htest(h_0)
    
    # Question 5
    htest = zscore(h_0, local_alignment[0])
    print 'mean: {:.3f} standard deviation: {:.3f} z-score: {:.3f}'.format(htest[0], htest[1], htest[2])

    
    
    
    
        