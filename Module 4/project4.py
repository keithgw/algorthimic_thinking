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
    
    scoring_matrix = {'-': {'-': dash_score}}
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
    
def compute_alignment_matrix(seq_x, seq_y, scoring_matrix, global_flag=True):
    """
    Inputs:
        seq_x, seq_y: two strings whose characters share a common alphabet
            with scoring_matrix
        scoring_matrix: output of build_scoring_matrix. Dictionary of 
            dictionaries whose [seq_x[i]][seq_y[j]] value is the score of the
            alignment of seq_x[i], seq_y[i].
        global_flag: if True, computes global alignment matrix. If false, 
            computes local alignment matrix.
    Output:
        A matrix whose ith, jth element is the maximum score for the alignment
        of the first i-1, j-1 elements of seq_x, seq_y.
    """
    alignment_matrix = [[0 for _m in range(len(seq_y) + 1)] 
                        for _n in range(len(seq_x) + 1)]
    
    # loop over zeroth-column, assign values for leading dashes on seq_y
    for idx in range(1, len(seq_x) + 1):
        gxval = alignment_matrix[idx - 1][0] + scoring_matrix[seq_x[idx - 1]]['-'] 
        if global_flag:
            alignment_matrix[idx][0] = gxval
        else:
            alignment_matrix[idx][0] = max(0, gxval) # local alignment
        
    # loop over zeroth-row, assign values for leading dashes on seq_x
    for jdx in range(1, len(seq_y) + 1):
        gyval = alignment_matrix[0][jdx - 1] + scoring_matrix['-'][seq_y[jdx - 1]]
        if global_flag:
            alignment_matrix[0][jdx] = gyval
        else:
            alignment_matrix[0][jdx] = max(0, gyval) # local alignment
    
    # loop over rest of sequence, assign max value from three possibilities:
    # -,Y; X,Y; X,-
    for idx in range(1, len(seq_x) + 1):
        for jdx in range(1, len(seq_y) + 1):
            x_y = scoring_matrix[seq_x[idx - 1]][seq_y[jdx - 1]]
            xdash = scoring_matrix[seq_x[idx - 1]]['-']
            dashy = scoring_matrix['-'][seq_y[jdx - 1]]
            if global_flag:
                alignment_matrix[idx][jdx] = max(alignment_matrix[idx - 1][jdx - 1] + x_y,
                                                 alignment_matrix[idx - 1][jdx] + xdash,
                                                 alignment_matrix[idx][jdx - 1] + dashy)
            else: # local alignment
                alignment_matrix[idx][jdx] = max(0,
                                                 alignment_matrix[idx - 1][jdx - 1] + x_y,
                                                 alignment_matrix[idx - 1][jdx] + xdash,
                                                 alignment_matrix[idx][jdx - 1] + dashy)
            
    
    return alignment_matrix
    
def compute_global_alignment(seq_x, seq_y, scoring_matrix, alignment_matrix):
    """
    Inputs:
        seq_x, seq_y: two character strings that share a common alphabet 
        with scoring_matrix
        scoring_matrix: output of build_scoring_matrix. Dictionary of 
            dictionaries whose [seq_x[i]][seq_y[j]] value is the score of the
            alignment of seq_x[i], seq_y[i].
        alignment_matrix: A matrix whose ith, jth element is the maximum score 
        for the alignment of the first i-1, j-1 elements of seq_x, seq_y.
    Outputs:
        a tuple (score, align_x, align_y), where score is the score of the 
        global alignment of align_x and align_y    
    """
    xdx, ydx = len(seq_x), len(seq_y)
    align_x, align_y = '', ''
    
    while xdx != 0 and ydx != 0:
        if alignment_matrix[xdx][ydx] == alignment_matrix[xdx - 1][ydx - 1] + scoring_matrix[seq_x[xdx - 1]][seq_y[ydx - 1]]:
            align_x = seq_x[xdx - 1] + align_x
            align_y = seq_y[ydx - 1] + align_y
            xdx -= 1
            ydx -= 1
        elif alignment_matrix[xdx][ydx] == alignment_matrix[xdx -1][ydx] + scoring_matrix[seq_x[xdx -1]]['-']:
            align_x = seq_x[xdx - 1] + align_x
            align_y = '-' + align_y
            xdx -= 1
        else:
            align_x = '-' + align_x
            align_y = seq_y[ydx - 1] + align_y
            ydx -= 1
    while xdx != 0:
        align_x = seq_x[xdx - 1] + align_x
        align_y = '-' + align_y
        xdx -= 1
    while ydx != 0:
        align_x = '-' + align_x
        align_y = seq_y[ydx - 1] + align_y
        ydx -=1
        
    return (alignment_matrix[len(seq_x)][len(seq_y)], align_x, align_y)