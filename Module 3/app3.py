# -*- coding: utf-8 -*-
""" 
Application 3 completed as part of Algorithmic Thinking offered by
Rice University on Coursera.
July 2015
Keith G. Williams
"""

# imports
import alg_cluster as clust
from project3 import slow_closest_pair
from project3 import fast_closest_pair
from project3 import hierarchical_clustering as hclust
from project3 import kmeans_clustering as kmeans
import random
from time import clock
import matplotlib.pyplot as plt
import gc
import numpy as np

"""Question 1
Write a function gen_random_clusters(num_clusters) that creates a list of 
clusters where each cluster in this list corresponds to one randomly generated 
point in the square with corners (±1,±1). Use this function and your favorite 
Python timing code to compute the running times of the functions 
slow_closest_pair and fast_closest_pair for lists of clusters of size 2 to 200.
Once you have computed the running times for both functions, plot the result as
two curves combined in a single plot. (Use a line plot for each curve.) The 
horizontal axis for your plot should be the the number of initial clusters
while the vertical axis should be the running time of the function in seconds.
Please include a legend in your plot that distinguishes the two curves.
"""

def gen_random_clusters(num_clusters):
    """creates a list of randomly generated clusters"""
    clusters = []
    for idx in range(num_clusters):
        clusters.append(clust.Cluster(set([]), 
                                      2 * random.random() - 1, 
                                      2 * random.random() - 1, 
                                      1, 0))
        
    return clusters

def timer(func, clusters):
    """timer function for comparing running times of functions for building
    targeted orders. Returns a list of time intervals."""
    delta_t = []
    
    gc.disable() # disable garbage collector for uninterupted timing    
    for clist in clusters:
        initial = clock()
        func(clist)
        final = clock()
        delta_t.append(final - initial)    

    gc.enable() # turn garbage collector back on
    return delta_t            

def q1():
    # generate random clusters
    clusters = []
    sizes = range(2, 201)
    for size in sizes:
        clusters.append(gen_random_clusters(size))
    
    # get running times
    random.seed(912)
    
    # run 10 trials, and take the median time for each n to smooth data
    slow_trials = np.zeros((10, 199))
    fast_trials = np.zeros((10, 199))
    for i in range(10):
        slow_trials[i,:] = timer(slow_closest_pair, clusters)
        fast_trials[i,:] = timer(fast_closest_pair, clusters)
       
    # times
    slow_times = np.median(slow_trials, 0)
    fast_times = np.median(fast_trials, 0)
    
    # plot
    plt.figure()
    plt.plot(sizes, slow_times, 'c-', label='slow_closest_pair')
    plt.plot(sizes, fast_times, 'm-', label='fast_closest_pair')
    plt.legend(loc='upper left')
    plt.xlabel('Size of Cluster List')
    plt.ylabel('Median Running Time (s), 10 Trials')
    plt.title('Comparison of Running Times on Desktop Python')
    plt.show()
    
    return None
    
if __name__ == '__main__':
    q1()