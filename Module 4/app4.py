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
    plt.hist(scores, normed=True, color='c', bins = 35)
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
    
def find_scoring_matrix(x, y, med, dim):
    """
    Find the scoring matrix that satisifes the definition of minimum edit
    distance: |x| + |y| - score(x, y)
    
    Inputs:
        x, y: english strings
        med: minimum edit distance between x, y
        dim: range of values to test for diag_score, off_score, dash_score
            note dash_scores will be <= 0
    """
    alphabet = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    
    # med(kitten, sitting) = 3
    correct = len(x) + len(y) - med # 10
    solutions = np.zeros((dim, dim, dim))
    for diag in range(dim):
        for off in range(dim):
            for dash in range(dim):
                sm = seq.build_scoring_matrix(alphabet, diag, off, -1 * dash)
                am = seq.compute_alignment_matrix(x, y, sm)
                solutions[diag, off, dash] = seq.compute_global_alignment(x, y, sm, am)[0]
    
    parameters = np.transpose(np.nonzero(solutions == correct))
    parameters[:, 2] *= -1
    return parameters
    
def check_spelling(checked_word, dist, word_list):
    """
    Returns a set of words from word_list that are dist edit distance from 
    checked_word
    """
    alphabet = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
    'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    candidates = set([])
        
    for word in word_list:
        smtrx = seq.build_scoring_matrix(alphabet, 2, 1, 0)
        amtrx = seq.compute_alignment_matrix(checked_word, word, smtrx)
        score = seq.compute_global_alignment(checked_word, word, smtrx, amtrx)[0]
        if len(checked_word) + len(word) - score <= dist:
            candidates.add(word)
                
    return candidates
        

def edits1(word):
    """
    Creates all 1-edit variants of checked_word.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    n = len(word)
    splits = [(word[:i], word[i:]) for i in range(n + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    substitutions = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    return set(deletes + inserts + substitutions)
                        
def fast_spell_check(checked_word, word_list):
    """
    Returns a set of words from word_list with edit distance <= 2 from
    checked_word.
    """
    # store word_list as a set for fast access
    corpus = set(word_list)
    
    # create all 1-edit variants of checked_word
    edits_1 = edits1(checked_word)
    
    # create all 1-edit variants of 1-edit variants of checked_word
    edits_2 = set([])
    for edit in edits_1:
        edits_1_1 = edits1(edit)
        edits_2 = edits_2.union(edits_1_1)
        
    edits = set([checked_word]).union(edits_1).union(edits_2)
    
    # return only the words from word_list that are in 0-, 1-, and 2-edit variants
    return edits.intersection(corpus)
    


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
    print 'Drosophila/Pax Percent:', pax_alignment[1] * 100, '%'
    print '=' * 30
    
    # Question 4
    random.seed(6022)
    NSIM = 1000
    h_0 = generate_null_distribution(human, drosophila, pam50, NSIM)
    plot_htest(h_0)
    
    # Question 5
    htest = zscore(h_0, local_alignment[0])
    print 'Hypothesis test:'
    print 'mean: {:.3f} standard deviation: {:.3f} z-score: {:.3f}'.format(htest[0], htest[1], htest[2])
    print '=' * 30
    
    # Question 6
    plot_htest(h_0)    
    
    # Question 7
    kitten_sitting = find_scoring_matrix('kitten', 'sitting', 3, 3)
    intention_execution = find_scoring_matrix('intention', 'execution', 5, 3)
    for solution in kitten_sitting:
        for _solution in intention_execution:
            if np.all(solution == _solution):
                print 'Solution to Scoring Matrix Paramters for Edit Distance:'
                print '[diag_score  off_diag_score  dash_score]'
                print solution
                print '=' * 30
                
    # Question 8
    word_list = read_words(WORD_LIST_URL)
    print 'Within 1 of "humble": \n'
    print check_spelling('humble', 1, word_list)
    print '=' * 30
    print 'Within 2 of "firefly": \n'
    print check_spelling('firefly', 2, word_list)
    print '=' * 30
    
    # Question 9
    print 'Within 1 of "humble" (fast): \n'
    print set(word_list).intersection(edits1('humble'))
    print '=' * 30
    print 'Within 2 of "firefly" (fast): \n'
    print fast_spell_check('firefly', word_list)
    
    
if __name__ == '__main__':
    run()
    