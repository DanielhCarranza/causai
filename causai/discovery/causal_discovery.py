"""Causal Discovery Algorithnms  """
import networkx as nx 
import pandas as pd 

class GES():
    """GES algorithm 
    **Description:** Greedy Equivalence Search algorithm. A score-based
    Bayesian algorithm that searches heuristically the graph which minimizes
    a likelihood score on the data.
    """

    def __init__(self, score='obs',verbose=None):
        """Init the model and its available arguments."""
    
        super(GES, self).__init__()
        self.scores = {'int': 'GaussL0penIntScore',
                       'obs': 'GaussL0penObsScore'}
        self.score = score

    def orient_undirected_graph(self, data, graph):
        """Run GES on an undirected graph.
        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.Graph): Skeleton of the graph to orient
        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.
        """
    def orient_directed_graph(self, data, graph):
        """Run GES on a directed graph.
        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.DiGraph): Skeleton of the graph to orient
        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.
        """
    def create_graph_from_data(self, data):
        """Run the GES algorithm.
        Args:
            data (pandas.DataFrame): DataFrame containing the data
        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.
        """
