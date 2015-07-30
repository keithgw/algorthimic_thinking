"""Project Module 4: Sequence Alignment
Completed as part of Algorithmic Thinking, offered by Rice University through
Coursera. August 2015"""

def build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score):
    """
    Inputs: 
        alphabet - A set of characters, which are members of a sequence.
        diag_score: score for alphabet[i] == alphabet[j]
        off_diag_score: score for alphabet[i] != alphabet[j]
        dash_score: score for matching a member of alphabet with a '-'
    Output:
        A dictionary of dictionaries indexed by members of alphabet + '-',
        whose value = score for the alignment of the keys.
    """
    
    scoring_matrix = {'-': {'-': 0}}
    for letter in alphabet:
        scoring_matrix['-'][letter] = dash_score
        scoring_matrix[letter] = {}
        for _letter in alphabet:
            if letter == _letter:
                scoring_matrix[letter][_letter] = diag_score
            else:
                scoring_matrix[letter][_letter] = off_diag_score
            scoring_matrix[letter]['-'] = dash_score
    
    return scoring_matrix