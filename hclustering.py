# -*- coding: utf-8 -*-
__author__ = 'Marco'

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from preprocessing import load_sparse_csr


X = load_sparse_csr()

# generate the linkage matrix
Z = linkage(X.toarray(),'ward')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Job')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=180.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
