from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

import argparse

parser = argparse.ArgumentParser(description="Graph similarity testing.")
parser.add_argument('--graph1', type=str, help='The name of the graph1 file')
parser.add_argument('--graph2', type=str, help='The name of the graph2 file')
# parser.add_argument('--chunk', type=int)

args = parser.parse_args()

# Assume that G1 and G2 are your two graphs
G1 = nx.read_edgelist(args.graph1, nodetype=int, data=(('weight',float),))
G2 = nx.read_edgelist(args.graph2, nodetype=int, data=(('weight',float),))

def get_top_n_vertices(graph, n):
    # Compute the degree of each vertex
    degrees = dict(graph.degree())
    # Sort the vertices by degree in descending order
    sorted_vertices = sorted(degrees, key=degrees.get, reverse=True)
    # Return the top n vertices
    return sorted_vertices[:n]

def get_induced_graph(graph, vertices):
    # Create a subgraph induced by the given vertices
    return graph.subgraph(vertices)

# Get the number of vertices in G2
n = len(G2.nodes)

# Get the top n vertices in G1
top_n_vertices_G1 = get_top_n_vertices(G1, n)

# Create the induced graph G3
G3 = get_induced_graph(G1, top_n_vertices_G1)

# nx.write_weighted_edgelist(G3, "graph_harry_induced_chunk"+str(args.chunk)+".txt")
nx.write_weighted_edgelist(G3, "graph_harry_induced.txt")

# Generate Node2Vec embeddings
node2vec = Node2Vec(G3, dimensions=64, walk_length=30, num_walks=200, workers=1)  # Set workers to 1
model1 = node2vec.fit(window=10, min_count=1, batch_words=4)

node2vec = Node2Vec(G2, dimensions=64, walk_length=30, num_walks=200, workers=1)  # Set workers to 1
model2 = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get the embeddings
embeddings1 = model1.wv.vectors
embeddings2 = model2.wv.vectors

# Compute the average embeddings
avg_embedding1 = np.mean(embeddings1, axis=0)
avg_embedding2 = np.mean(embeddings2, axis=0)

# Compute the cosine similarity between the average embeddings
similarity = cosine_similarity([avg_embedding1], [avg_embedding2])

print(similarity[0][0])