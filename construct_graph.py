# git clone https://github.com/shtoshni92/long-doc-coref.git
# pip install transformers
# pip install spacy
# python -m spacy download en_core_web_lg

# ====== RULL ALL THE ABOVE COMMANDS IN THE TERMINAL BEFORE DOING ANYTHING ELSE ======

import os
import sys
sys.path.append('long-doc-coref/src')

import argparse

parser = argparse.ArgumentParser(description="Perform coreference resolution and construct graph.")
parser.add_argument('--filename', type=str, help='The name of the file')
parser.add_argument('--outfile', type=str, help='The name of the output file')
parser.add_argument('--window', type=int, help='The interaction sliding window size')
parser.add_argument('--factor', type=float, help='Division factor', default=1)
args = parser.parse_args()

# This will also download the SpanBERT model finetuned for Coreference (by Joshi et al, 2020) from Huggingface
from inference.inference import Inference
inference_model = Inference("litbank_lbmem_model.pth")

with open(args.filename, "r") as text_file:
   doc = text_file.read()
   
output = inference_model.perform_coreference(doc)

import spacy

# Load the large English model
nlp = spacy.load('en_core_web_lg')

# Process the text
res = nlp(doc)

person_list = []

# Iterate over the entities
for ent in res.ents:
    # Print the entity text and its label
    if ent.label_ == "PERSON":
        # print(ent.text, ent.label_)
        person_list.append(ent.text)
        
# print(person_list)

person_clusters = []

for cluster in output["clusters"]:
    for p in cluster:
        # print(p)
        # # print(p[1])
        if p[1] in person_list:
            person_clusters.append(cluster)
            break

elems_with_idx = []
idx = 0
for p in person_clusters:
    # print(p)
    for e in p:
        temp = (idx, e[0], e[1])
        elems_with_idx.append(temp)
    idx += 1
        
print(elems_with_idx)   

sorted_elems_with_idx = sorted(elems_with_idx, key=lambda x: x[1][0])
print(sorted_elems_with_idx)

from collections import defaultdict 
  
# function for adding edge to graph 
graph = defaultdict(lambda: defaultdict(int)) 
def addEdge(graph,u,v):
    if v not in graph[u]:
        graph[u][v] = 1
    else:
        graph[u][v] += 1
  
# definition of function 
def generate_edges(graph):
    open(args.outfile, "w").close 
    edges = [] 
    # for each node in graph 
    for node in graph: 
        # for each neighbour node of a single node 
        for neighbour, weight in graph[node].items():     
            # if edge exists then append 
            with open(args.outfile, "a") as f:
                f.write(str(node) + " " + str(neighbour) + " " + str(weight/args.factor) + "\n")
            edges.append((node, neighbour, weight)) 
    return edges 

interaction_sliding_window_size = args.window

for i in range(1, len(sorted_elems_with_idx)):
    if sorted_elems_with_idx[i][0] == sorted_elems_with_idx[i-1][0]:
        continue
        # discarding self edges
    curr_start = sorted_elems_with_idx[i][1][0]
    prev_end = sorted_elems_with_idx[i-1][1][1]
    if curr_start < prev_end+interaction_sliding_window_size:
        addEdge(graph, sorted_elems_with_idx[i-1][0], sorted_elems_with_idx[i][0])
        addEdge(graph, sorted_elems_with_idx[i][0], sorted_elems_with_idx[i-1][0])
  
# Driver Function call  
# to print generated graph 
print(generate_edges(graph))  
        
    