from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self):
        self._graph = nx.DiGraph()

    def add_actor(self, actor):
        """
        Add an actor to the graph.
        
        Parameters:
        actor (Actor): The actor to add to the graph.
        """
        self._graph.add_node(actor.name)

    def add_transaction(self, transaction):
        """
        Add a transaction to the graph.
        
        Parameters:
        transaction (Transaction): The transaction to add to the graph.
        """
        self._graph.add_edge(
            transaction.source.name,
            transaction.target.name,
            value=transaction.value,
            nb_transactions=transaction.nb_transactions,
            date=transaction.date,
        )
    
    def get_actor_neighbors(self, actor_name):
        """
        Get the neighbors (adjacent nodes) of a specific actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        list: List of neighbor actor names.
        """
        return list(self._graph.neighbors(actor_name))

    def get_actor_in_degree(self, actor_name):
        """
        Get the in-degree (number of incoming edges) of a specific actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        int: In-degree of the actor.
        """
        return self._graph.in_degree(actor_name)

    def get_actor_out_degree(self, actor_name):
        """
        Get the out-degree (number of outgoing edges) of a specific actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        int: Out-degree of the actor.
        """
        return self._graph.out_degree(actor_name)

    def get_shortest_path(self, source_actor, target_actor):
        """
        Find the shortest path between two actors.
        
        Parameters:
        source_actor (str): Name of the source actor.
        target_actor (str): Name of the target actor.
        
        Returns:
        list: List of actor names representing the shortest path.
        """
        try:
            return nx.shortest_path(self._graph, source_actor, target_actor)
        except nx.NetworkXNoPath:
            return None
        
    def get_subgraph(self, actor_names):
        """
        Get a subgraph of the graph.
        
        Parameters:
        actor_names (list): List of actor names.
        
        Returns:
        Graph: The subgraph.
        """
        subgraph = self._graph.subgraph(actor_names)
        
        subgraph_instance = Graph()
        subgraph_instance._graph = subgraph
        
        return subgraph_instance
        
    def get_edges_subgraph(self, actor_name, in_edges=False, out_edges=False):
        """
        Get a subgraph with all the incoming edges for a specific actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        Graph: The subgraph with incoming edges.
        """
        if in_edges and out_edges:
            edges_in = list(self._get_in_edges(actor_name))
            edges_out = list(self._get_out_edges(actor_name))
            edges = edges_in + edges_out 
        elif in_edges:
            edges = list(self._get_in_edges(actor_name))
        elif out_edges:
            edges = list(self._get_out_edges(actor_name))
        else:
            raise ValueError("You must specify either in_edges or out_edges.")

        # Extracting edge data (attributes) for the subgraph
        edge_data = {(source, target): data for source, target, data in edges}

        # Creating a new subgraph with the specified edges
        edges_subgraph = nx.DiGraph()

        # Adding nodes from edges and updating edge attributes
        for (source, target), data in edge_data.items():
            edges_subgraph.add_edge(source, target, **data)

        subgraph_instance = Graph()
        subgraph_instance._graph = edges_subgraph
        
        return subgraph_instance
    
    def _get_out_edges(self, actor_name):
        """
        Get all the outgoing edges for a specific actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        list: List of outgoing edges.
        """
        return self._graph.out_edges(actor_name, data=True)
    
    def _get_in_edges(self, actor_name):
        """
        Get all the incoming edges for a specific actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        list: List of incoming edges.
        """
        return self._graph.in_edges(actor_name, data=True)
    
    def get_graph(self):
        """
        Get the graph.
        
        Returns:
        nx.Graph: The graph.
        """
        return self._graph

    def plot_graph(self):
        pos = nx.spring_layout(self._graph)
        nx.draw(self._graph, pos, with_labels=True, node_color='skyblue', font_size=8, font_color='black')
        plt.show()