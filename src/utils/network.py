from networkx.algorithms.community import greedy_modularity_communities
import community as community_louvain
from utils.community import Community
import matplotlib.pyplot as plt
from utils.graph import Graph
import networkx as nx    
import tabulate
from tabulate import tabulate

class Network(Graph):
    def __init__(self, actors):
        super().__init__()
        self.actors = actors
        self.communities = []
    
    def construct_network(self):
        for actor in self.actors:
            self.add_actor(actor)
            
        for actor in self.actors:
            for transaction in actor.transactions:
                self.add_transaction(transaction)
                
    def process_communities_girvan_newman(self):
        self.communities = list(greedy_modularity_communities(self._graph))
                
    def process_communities_louvain(self):
        undirected_graph = self._graph.to_undirected()
        partition = community_louvain.best_partition(undirected_graph)
        self.communities = [[node for node, com_id in partition.items() if com_id == community_id] for community_id in set(partition.values())]

    def get_communities(self):
        """
        Get the communities of the network.
        
        Returns:
        list: List of communities.
        """
        return self.communities

    def get_community(self, actor_name):
        """
        Get the community to which an actor belongs.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        set: The community to which the actor belongs.
        """
        for community in self.communities:
            if actor_name in community:
                return community
            
        return None

    def get_community_graph(self, actor_name):
        """
        Get the community graph of an actor.
        
        Parameters:
        actor_name (str): Name of the actor.
        
        Returns:
        Graph: The community graph.
        """
        community = self.get_community(actor_name)
        community_graph = self._graph.subgraph(community)
        
        community_graph_instance = Graph()
        community_graph_instance._graph = community_graph
        
        return community_graph_instance
        
    def print_communities(self):
        """
        Print the communities.
        """
        table = [["Id", "Members"]]
        
        for community_id, members in enumerate(self.communities):
            table.append([community_id, sorted(members)])

        print(tabulate(table, headers="firstrow"))