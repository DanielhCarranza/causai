"""Utility functions for causai module."""
import os
import numpy as np

from tqdm import tqdm
from typing import Union
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from concurrent.futures import as_completed, ThreadPoolExecutor

import networkx as nx 
import pandas as pd
import operator


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  
        self.update(blocks * bsize - self.n)  


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def download_urls(urls, filenames):
    """Download urls to filenames in a multi-threaded way."""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(urlretrieve, url, filename) for url, filename in zip(urls, filenames)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:  
                print("Error", e)

def read_list_edges(filename, directed=True, **kwargs):
    """Read a file (containing list of edges) and convert it into a directed
    or undirected networkx graph.
    :param filename: file to read or DataFrame containing the data
    :type filename: str or pandas.DataFrame
    :param directed: Return directed graph
    :type directed: bool
    :param kwargs: extra parameters to be passed to pandas.read_csv
    :return: networkx graph containing the graph.
    :rtype: **networkx.DiGraph** or **networkx.Graph** depending on the
      ``directed`` parameter.
    Examples:
        >>> from cdt.utils import read_adjacency_matrix
        >>> data = read_causal_pairs('graph_file.csv', directed=False)
    """
    if isinstance(filename, str):
        data = pd.read_csv(filename, **kwargs)
    elif isinstance(filename, pd.DataFrame):
        data = filename
    else:
        raise TypeError("Type not supported.")
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    if len(data.columns) == 3:
        data.columns = ['Cause', 'Effect', 'Score']
    else:
        data.columns = ['Cause', 'Effect']

    for idx, row in data.iterrows():
        try:
            score = row["Score"]
        except KeyError:
            score = 1
        graph.add_edge(row['Cause'], row["Effect"], weight=score)

    return graph

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