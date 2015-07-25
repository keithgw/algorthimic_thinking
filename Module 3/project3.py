"""
Student template code for Project 3
Student will implement five functions:

slow_closest_pair(cluster_list)
fast_closest_pair(cluster_list)
closest_pair_strip(cluster_list, horiz_center, half_width)
hierarchical_clustering(cluster_list, num_clusters)
kmeans_clustering(cluster_list, num_clusters, num_iterations)

where cluster_list is a 2D list of clusters in the plane
"""

import math
import alg_cluster



######################################################
# Code for closest pairs of clusters

def pair_distance(cluster_list, idx1, idx2):
    """
    Helper function that computes Euclidean distance between two clusters in a list

    Input: cluster_list is list of clusters, idx1 and idx2 are integer indices for two clusters
    
    Output: tuple (dist, idx1, idx2) where dist is distance between
    cluster_list[idx1] and cluster_list[idx2]
    """
    return (cluster_list[idx1].distance(cluster_list[idx2]), min(idx1, idx2), max(idx1, idx2))


def slow_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (slow)

    Input: cluster_list is the list of clusters
    
    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.       
    """
    
    min_distance = set([(float('inf'), 0, 0)])

    for num_idx1 in xrange(len(cluster_list)):
        _node_idx1 = cluster_list[num_idx1]
        num_idx2 = num_idx1 + 1

        for _node_idx2 in cluster_list[num_idx2:]:
            new_distance = pair_distance(cluster_list, num_idx1, num_idx2)

            if list(new_distance)[0] < list(min_distance)[0][0]:
                min_distance = set([new_distance])
            elif list(new_distance)[0] == list(min_distance)[0][0]:
                min_distance.add(new_distance)
            num_idx2 += 1

    return min_distance.pop()


def fast_closest_pair(cluster_list):
    """
    Compute the distance between the closest pair of clusters in a list (fast)

    Input: cluster_list is list of clusters SORTED such that horizontal positions of their
    centers are in ascending order
    
    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] have minimum distance dist.       
    """
    
    def fast_helper(clist, h_order, v_order):
        """
        Divide and conquer method for computing distance between closest pair of points
        Running time is O(n * log(n))
        h_order and v_order are lists of indices for clusters
        ordered horizontally and vertically
        Returns a tuple (distance, idx1, idx2) with idx1 < idx 2 where
        clist[idx1] and clist[idx2]
        have the smallest distance dist of any pair of clusters
        """
        def _div(h_order):
            """
            divide
            """
            return int(math.ceil(len(h_order) / 2.0))

        # base case
        if len(h_order) <= 3:
            sublist = [clist[h_order[i]]
                    for i in range(len(h_order))]
            res = slow_closest_pair(sublist)
            return res[0], h_order[res[1]], h_order[res[2]]

        # divide
        mid = 0.5 * (clist[h_order[_div(h_order) - 1]].horiz_center() +
                     clist[h_order[_div(h_order)]].horiz_center())

        _hlr = h_order[0: _div(h_order)], h_order[_div(h_order): len(h_order)]
        min_d = min(fast_helper(clist, _hlr[0],
            [vi for vi in v_order if vi in frozenset(_hlr[0])]),
            fast_helper(clist, _hlr[1],
                [vi for vi in v_order if vi in frozenset(_hlr[1])]))

        # conquer
        sss = [vi for vi in v_order if
                abs(clist[vi].horiz_center() - mid) < min_d[0]]

        for _uuu in range(len(sss) - 1):
            for _vvv in range(_uuu + 1, min(_uuu + 4, len(sss))):
                dsuv = clist[sss[_uuu]].distance(clist[sss[_vvv]])
                min_d = min((min_d), (dsuv, sss[_uuu], sss[_vvv]))

        return min_d[0], min(min_d[1], min_d[2]), max(min_d[1], min_d[2])

    # compute list of indices for the clusters ordered in the horizontal direction
    hcoord_and_index = [(cluster_list[idx].horiz_center(), idx)
            for idx in range(len(cluster_list))]
    #  print hcoord_and_index
    hcoord_and_index.sort()
    #  print hcoord_and_index
    horiz_order = [hcoord_and_index[idx][1]
            for idx in range(len(hcoord_and_index))]

    # compute list of indices for the clusters ordered in vertical direction
    vcoord_and_index = [(cluster_list[idx].vert_center(), idx)
            for idx in range(len(cluster_list))]
    vcoord_and_index.sort()
    vert_order = [vcoord_and_index[idx][1]
            for idx in range(len(vcoord_and_index))]

    # compute answer recursively
    #  print vert_order[0].real
    fast_helper(cluster_list, horiz_order, vert_order)
    answer = fast_helper(cluster_list, horiz_order, vert_order)
    #  return slow_closest_pairs(cluster_list)
    return (answer[0], min(answer[1:]), max(answer[1:]))


def closest_pair_strip(cluster_list, horiz_center, half_width):
    """
    Helper function to compute the closest pair of clusters in a vertical strip
    
    Input: cluster_list is a list of clusters produced by fast_closest_pair
    horiz_center is the horizontal position of the strip's vertical center line
    half_width is the half the width of the strip (i.e; the maximum horizontal distance
    that a cluster can lie from the center line)

    Output: tuple of the form (dist, idx1, idx2) where the centers of the clusters
    cluster_list[idx1] and cluster_list[idx2] lie in the strip and have minimum distance dist.       
    """

    # compute list of indices for the clusters ordered in the horizontal direction
    strip_and_index = [(cluster_list[idx], idx)
            for idx in range(len(cluster_list)) 
            if abs(cluster_list[idx].horiz_center() - horiz_center) < half_width]
    #print 'strip_and_index:', strip_and_index
    #hcoord_and_index.sort()
    #  print hcoord_and_index
    #horiz_order = [hcoord_and_index[idx][1]
    #        for idx in range(len(hcoord_and_index))]
    
    # compute list of indices for the clusters ordered in vertical direction
    vcoord_and_index = [(strip_and_index[idx][0].vert_center(), idx)
            for idx in range(len(strip_and_index))]
    vcoord_and_index.sort()
    vert_order = [strip_and_index[vcoord_and_index[idx][1]][1]
            for idx in range(len(vcoord_and_index))]     
    #print 'vert_order:', vert_order       
    
    dij = (float('inf'), -1, -1)
    for udx in range(len(vert_order) - 1):    # len - 2?
        start = udx + 1
        stop = min(udx + 4, len(vert_order))
        for vdx in range(start, stop):
            dist = pair_distance(cluster_list, vert_order[udx], vert_order[vdx])
            #print 'udx, vdx:', udx, vdx, 'dist:', dist
            if dist[0] < dij[0]:
                dij = dist
    
    return dij
        
#### TESTING ####  
#print closest_pair_strip(([alg_cluster.Cluster(set([]), -4.0, 0.0, 1, 0), 
#alg_cluster.Cluster(set([]), 0.0, -1.0, 1, 0), 
#alg_cluster.Cluster(set([]), 0.0, 1.0, 1, 0), 
#alg_cluster.Cluster(set([]), 4.0, 0.0, 1, 0)]), 0.0, 4.1231059999999999)
## END TESTING ##

######################################################################
# Code for hierarchical clustering


def hierarchical_clustering(cluster_list, num_clusters):
    """
    Compute a hierarchical clustering of a set of clusters
    Note: the function may mutate cluster_list
    
    Input: List of clusters, integer number of clusters
    Output: List of clusters whose length is num_clusters
    """
    
    new_cluster_list = list(cluster_list)

    while len(new_cluster_list) > num_clusters:
        _, node1, node2 = fast_closest_pair(new_cluster_list)
        new_cluster_list[node1].merge_clusters(new_cluster_list[node2])
        del new_cluster_list[node2]

    return new_cluster_list

######################################################################
# Code for k-means clustering

    
def kmeans_clustering(cluster_list, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters
    Note: the function may not mutate cluster_list
    
    Input: List of clusters, integers number of clusters and number of iterations
    Output: List of clusters whose length is num_clusters
    """

    # position initial clusters at the location of clusters with largest populations
            
    cluster_n = len(cluster_list)

    miu_k = sorted(cluster_list,
              key=lambda c: c.total_population())[-num_clusters:]
    miu_k = [c.copy() for c in miu_k]

    # n: cluster_n
    # q: num_iterations
    for _ in xrange(num_iterations):
        cluster_result = [alg_cluster.Cluster(set([]), 0, 0, 0, 0) for _ in range(num_clusters)]
        # put the node into closet center node

        for jjj in xrange(cluster_n):
            min_num_k = 0
            min_dist_k = float('inf')
            for num_k in xrange(len(miu_k)):
                dist = cluster_list[jjj].distance(miu_k[num_k])
                if dist < min_dist_k:
                    min_dist_k = dist
                    min_num_k = num_k

            cluster_result[min_num_k].merge_clusters(cluster_list[jjj])

        # re-computer its center node
        for kkk in xrange(len(miu_k)):
            miu_k[kkk] = cluster_result[kkk]

    return cluster_result

