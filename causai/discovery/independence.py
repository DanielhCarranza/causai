""" Independece Tests """

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
import operator

class Glasso():
    def __init__(self):
        super(Glasso, self).__init__()
    def predict(self, data:pd.DataFrame, alpha:float=0.01, max_iter:int=2000, **kwargs)->nx.Graph:
        """Predict the graph structure """
        edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(data.values)
        return nx.relabel_nodes(nx.DiGraph(edge_model.get_precision()), 
                            {idx: i for idx, i in enumerate(data.columns)})

def aracne(m, **kwargs):
    """Implementation of the ARACNE algorithm.
    .. note::
       For networkx graphs, use the cdt.utils.graph.remove_indirect_links function
    Args:
        mat (numpy.ndarray): matrix, if it is a square matrix, the program assumes
            it is a relevance matrix where mat(i,j) represents the similarity content
            between nodes i and j. Elements of matrix should be
            non-negative.
    Returns:
        numpy.ndarray: Output deconvolved matrix (direct dependency matrix). Its components
        represent direct edge weights of observed interactions.
    Example:
        >>> from cdt.utils.graph import aracne
        >>> import networkx as nx
        >>> # Generate sample data
        >>> from cdt.data import AcyclicGraphGenerator
        >>> graph = AcyclicGraphGenerator(linear).generate()[1]
        >>> adj_mat = nx.adjacency_matrix(graph).todense()
        >>> output = aracne(adj_mat)
    .. note::
       Ref: ARACNE: An Algorithm for the Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context
       Adam A Margolin, Ilya Nemenman, Katia Basso, Chris Wiggins, Gustavo Stolovitzky, Riccardo Dalla Favera and Andrea Califano
       DOI: https://doi.org/10.1186/1471-2105-7-S1-S7
    """
    I0 = kwargs.get('I0', 0.0)  # No default thresholding
    W0 = kwargs.get('W0', 0.05)

    # thresholding
    m = np.where(m > I0, m, 0)

    # Finding triplets and filtering them
    for i in range(m.shape[0]-2):
        for j in range(i+1, m.shape[0]-1):
            for k in range(j+1, m.shape[0]):
                triplet = [m[i, j], m[j, k], m[i, k]]
                min_index, min_value = min(enumerate(triplet), key=operator.itemgetter(1))
                if 0 < min_value < W0:
                    if min_index == 0:
                        m[i, j] = m[j, i] = 0.
                    elif min_index == 1:
                        m[j, k] = m[k, j] = 0.
                    else:
                        m[i, k] = m[k, i] = 0.
    return m

def remove_indirect_links(g, **kwargs):
    """Apply deconvolution to a networkx graph.
    Args:
       g (networkx.Graph): Graph to apply deconvolution to
       alg (str): Algorithm to use ('aracne', 'clr', 'nd')
       kwargs (dict): extra options for algorithms
    Returns:
       networkx.Graph: graph with undirected links removed.

    """
    alg = aracne
    order_list = list(g.nodes())
    mat = np.array(nx.adjacency_matrix(g, nodelist=order_list).todense())
    return nx.relabel_nodes(nx.DiGraph(alg(mat, **kwargs)),
                            {idx: i for idx, i in enumerate(order_list)})