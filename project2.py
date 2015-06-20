"""
Project 2

Completed as part of Algorithmic Thinking offered by Rice University through
Coursera, May - Aug 2015.

Keith G. Williams

- Implementation of Breadth First Search (BFS)
- Computation of the set of connected components (CC) of an undirected graph
  and determination of the largest CC.
- Graph resilience measured by the size of the largest CC as a sequence of 
  nodes are deleted from a graph.
"""

# imports
from collections import deque

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
        if node in ugraph.keys():
            for edge in ugraph[node]:
                ugraph[edge].remove(node)
            del ugraph[node]
            resilience.append(largest_cc_size(ugraph))
        else:
            print 'error: node ', node, ' not in graph.'
        
    return resilience