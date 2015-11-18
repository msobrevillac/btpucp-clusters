# -*- coding: utf-8 -*-
__author__ = 'Fernando'

from preprocessing import load_sparse_csr, load_labels
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd



def visualize_clusters(tfidf_matrix, vocabulary, km):

    # calcuate the cosine distance between each document
    # this will be used for plotting on a euclidean (2-dimensional) plane.
    dist = 1 - cosine_similarity(tfidf_matrix)
    clusters = km.labels_.tolist()

    # convert two components as we are plotting points in a two-dimensional plane
    # 'precomputed' because we provide a distance matrix
    # we will also specify 'random_state' so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]

    # set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#99cc00'}

    # set up cluster names using a dict (perhaps using the top terms of each cluster)
    cluster_names = {0: '0',
                     1: '1',
                     2: '2',
                     3: '3',
                     4: '4',
                     5: '5'}

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

    #group by cluster
    groups = df.groupby('label')


    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',         # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',        # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

    plt.show() #show the plot

    # plt.close()


def main():
    # read the preprocessed data
    tfidf_matrix = load_sparse_csr()
    vocabulary = load_labels()

    # k-means clustering
    num_clusters = 6
    km = KMeans(n_clusters=num_clusters, n_jobs=-1)
    km.fit(tfidf_matrix)

    # visualize the generated clusters
    visualize_clusters(tfidf_matrix, vocabulary, km)


main()
