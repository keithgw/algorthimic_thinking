"""Project 1 completed as part of Algorithmic Thinking on Coursera from Rice
University. May 27, 2015 run of the course"""

## Part 1: Representing directed graphs

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
     
    # loop over nodes and set values for each key       
    for node in nodes:
        outs = set(nodes)
        outs.remove(node)
        graph[node] = outs
        
    return graph

## Part 2: Computing degree distributions

def compute_in_degrees(digraph):
    """Takes a directed graph represented as a dictionary and computes the 
    in-degrees for the nodes in the graph. Returns a dictionary with the same 
    set of keys (nodes) as digraph whose corresponding values are the number of
    edges whose head matches a particular node."""
    
    # define empty dictionary
    in_degrees = {}
    
    # define keys as nodes in digraph
    nodes = [key for key in digraph]
    
    # loop over nodes and count number of times each node appears in the
    # values of other nodes in digraph
    for node in nodes:
        count = 0
        for ndoe in nodes:
            if node in digraph[ndoe]:
                count += 1
        in_degrees[node] = count
    
    return in_degrees
    
def in_degree_distribution(digraph):
    """Takes a directed graph represented as a dictionary and comptes the
    unnormalized distribution of the in-degrees of the graph. Returns a 
    dictionary whose keys correspond to in-degrees of nodes in the graph. The
    value associated with each particular in-degree is the number of nodes with
    that in-degree. In-degrees with no corresponding nodes in the graph are not
    included in the dictionary."""
    
    # compute the in degree for each node in the directed graph
    in_degrees = compute_in_degrees(digraph)
    
    # loop over nodes, add value to distribution dictionary
    distro = {}
    for node in in_degrees:
        if in_degrees[node] in distro:
            distro[in_degrees[node]] += 1
        else:
            distro[in_degrees[node]] = 1
    
    return distro
    
"""
Provided code for Application portion of Module 1

Imports physics citation graph 
"""

# general imports
import urllib2

##################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph

citation_graph = load_graph(CITATION_URL)