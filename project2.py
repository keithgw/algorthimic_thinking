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
    Inputs
        ugraph: undirected graph represented as an adjacency list
    Output
        list of sets of nodes in a connected component
    """
    # initialize
    remaining_nodes = ugraph.keys()
    connected_components = list()
    
    while remaining_nodes:
        node = remaining_nodes[0]
        visited = bfs_visited(ugraph, node)
        connected_components.append(visited)
        for visited_node in visited:
            remaining_nodes.remove(visited_node)
            
    return connected_components
    
### TESTING ###
#EX_GRAPH = {0: set([1, 4, 5]), 
#1: set([0, 2, 5]), 
#2: set([1, 3, 5]), 
#3: set([0, 2]), 
#4: set([1]), 
#5: set([0, 1, 2]),
#6: set([])}
#
#print cc_visited(EX_GRAPH)