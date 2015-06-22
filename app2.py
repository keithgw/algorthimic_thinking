"""
Application 2

Completed as part of Algorithmic Thinking offered by Rice University through
Coursera, May - Aug 2015.

Keith G. Williams
"""

# general imports
import urllib2
import random
import time
import math
import matplotlib.pyplot as plt
from collections import deque


############################################
# Provided code

class UPATrial:
    """
    Simple class to encapsulate optimizated trials for the UPA algorithm
    
    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that each node number
        appears in correct ratio
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order
    


##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


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

##########################################################
# Code for BFS, computing CCs, largest CC size, and resilience
            
def bfs_visited(ugraph, start_node):
    """
    Inputs
        ugraph: undirected graph represented as an adjacency list
        start_node: node to which all connected components will be computed
    Output
        set of all nodes that are visited by BFS starting at start_node
    """
    # initialize queue and visited (CC of start_node)
    queue = deque()
    visited = set([start_node])
    queue.append(start_node)
    
    while queue:
        node = queue.popleft()
        for neighbor in ugraph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
    
def cc_visited(ugraph):
    """
    Input
        ugraph: undirected graph represented as an adjacency list
    Output
        list of sets of nodes in a connected component
    """
    # initialize
    remaining_nodes = ugraph.keys()
    connected_components = []
    
    while remaining_nodes:
        node = remaining_nodes[0]
        visited = bfs_visited(ugraph, node)
        connected_components.append(visited)
        for visited_node in visited:
            remaining_nodes.remove(visited_node)
            
    return connected_components

def largest_cc_size(ugraph):
    """
    Input
        ugraph: undirected graph represented as an adjacency list
    Output
        integer representing the size (number of nodes) 
        of the largest connected component in ugraph.
    """
    if ugraph:
        return max([len(cc) for cc in cc_visited(ugraph)])
    else:
        return 0

def compute_resilience(ugraph, attack_order):
    """
    Inputs
        ugraph: undirected graph represented as an adjacency list
        attack_order: a list of nodes in ugraph
    Output
        list whose k + 1th entry is the size of the largest connected
        component in the graph after the removal of the first k nodes in
        attack_order. The first entry is the size of the largest connected
        component in ugraph.
        
        For each node in attack_order, the function removes the given node
        and its edges from the graph and then computes the size of the 
        largest connected component for the resulting graph.
    """
    # initialize
    resilience = [largest_cc_size(ugraph)]
    
    for node in attack_order:
        # check that node exists
        if node in ugraph.keys():
            delete_node(ugraph, node)     # will also deleted edges
            # add largest remaining cc to resilience list
            resilience.append(largest_cc_size(ugraph))
        else:
            print 'error: node ', node, ' not in graph.'
        
    return resilience
    
##########################################################
# Code for implementing ER graphs and UPA graphs

def gen_er_ugraph(n, p):
    """generates a random, directed graph with n nodes, and probaility p
    of creating an edge from the ith to jth node. n_nodes must be an 
    integer, self-loops are not allowed."""
    
    # define undirected graph
    ugraph = {}
    
    # special case n = 0 (empty graph)
    if n == 0:
        return ugraph
    
    # for all n > 0
    else:
        # loop over each node
        for node in range(0, n):
            # loop over each potential adjacent node
            if node not in ugraph:
                ugraph[node] = set()
            for neighbor in range(0, n):
                if neighbor != node:             # no self-loops
                    if neighbor not in ugraph:
                        ugraph[neighbor] = set()
                    a = random.random()
                    if a < p:
                        ugraph[node].add(neighbor)
                        ugraph[neighbor].add(node)
        return ugraph

def make_complete_ugraph(num_nodes):
    """Takes the number of nodes and returns a dictionary corresponding to a 
    complete undirected graph with the specified number of nodes. A complete 
    graph contains all possible edges subject to the restriction that 
    self-loops are not allowed."""
    
    # define empty graph
    ugraph = {}
    
    # define list of nodes
    nodes = range(0, num_nodes)
     
    # loop over nodes and set values for each key       
    for node in nodes:
        outs = set(nodes)
        outs.remove(node)
        ugraph[node] = outs
        
    return ugraph
    
def upa(n, m):
    """Takes as an input number of nodes (n), where n > 0, and an integer 
    m, where 0 < m <= n. m is the number of existing nodes to which each
    new node will be randomly connected. Outputs an undirected graph"""
    
    # make a complete graph on m nodes
    ugraph = make_complete_ugraph(m)
    upa_trial = UPATrial(m)
    
    # add n - m nodes
    for i in range(m, n):
        ugraph[i] = upa_trial.run_trial(m)
        for neighbor in ugraph[i]:
            ugraph[neighbor].add(i)
    
    return ugraph

