""" Independece Tests """

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso

class Glasso():
    def __init__(self):
        super(Glasso, self).__init__()
    def predict(self, data:pd.DataFrame, alpha:float=0.01, max_iter:int=2000, **kwargs)->nx.Graph:
        """Predict the graph structure """
        edge_model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        edge_model.fit(data.values)
        return nx.relabel_nodes(nx.DiGraph(edge_model.get_precision()), 
                            {idx: i for idx, i in enumerate(data.columns)})

