"""Project 1 completed as part of Algorithmic Thinking on Coursera from Rice
University. May 27, 2015 run of the course"""

# Define directed graphs as dictionaries
EX_GRAPH0 = {0: set([1, 2]), 1: set([]), 2: set([])}

EX_GRAPH1 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3]), 3: set([0]), 
4: set([1]), 5: set([2]), 6: set([])}

EX_GRAPH2 = {0: set([1, 4, 5]), 1: set([2, 6]), 2: set([3, 7]), 3: set([7]),
4: set([1]), 5: set([2]), 6: set([]), 7: set([3]), 8: set([1, 2]), 
9: set([0, 3, 4, 5, 6, 7])}

def make_complete_graph(num_nodes):
    """Takes the number of nodes and returns a dictionary corresponding to a 
    complete directed graph with the specified number of nodes. A complete 
    graph contains all possible edges subject to the restriction that 
    self-loops are not allowed."""
    
    # define empty graph
    graph = {}
    
    # define list of nodes
    nodes = range(0, num_nodes)
        
    for node in nodes:
        outs = set(nodes)
        outs.remove(node)
        graph[node] = outs
        
    return graph