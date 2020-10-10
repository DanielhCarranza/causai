""" Vizualizations of graphs, metrics and data  """


import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(graph:nx.DiGraph, **kwargs):
    nx.draw(graph, with_labels=True, font_weight='bold',font_color='red')