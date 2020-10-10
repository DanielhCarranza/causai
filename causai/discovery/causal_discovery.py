"""Causal Discovery Algorithnms  """
import networkx as nx 

class GES():
    """GES algorithm **[R model]**.
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
        # Building setup w/ arguments.
        fe = DataFrame(nx.adj_matrix(graph, weight=None).todense())
        fg = DataFrame(1 - fe.values)

        results = self._run_ges(data, fixedGaps=fg, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})

    def orient_directed_graph(self, data, graph):
        """Run GES on a directed graph.
        Args:
            data (pandas.DataFrame): DataFrame containing the data
            graph (networkx.DiGraph): Skeleton of the graph to orient
        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.
        """
        return self.orient_undirected_graph(data, nx.Graph(graph))

    def create_graph_from_data(self, data):
        """Run the GES algorithm.
        Args:
            data (pandas.DataFrame): DataFrame containing the data
        Returns:
            networkx.DiGraph: Solution given by the GES algorithm.
        """

        results = self._run_ges(data, verbose=self.verbose)

        return nx.relabel_nodes(nx.DiGraph(results),
                                {idx: i for idx, i in enumerate(data.columns)})
