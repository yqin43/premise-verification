import networkx as nx
import igraph as ig
import leidenalg


G = nx.karate_club_graph()  # graph


def convert_networkx_to_igraph(nx_graph):
    ig_graph = ig.Graph(directed=False)
    ig_graph.add_vertices(nx_graph.nodes())
    ig_graph.add_edges(nx_graph.edges())
    
    # ig_graph.es['weight'] = [nx_graph[u][v]['weight'] for u, v in nx_graph.edges()]
    
    return ig_graph

# Convert G
ig_G = convert_networkx_to_igraph(G)

# Apply the Leiden algorithm
partition = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition)

# Convert partition to a format usable in networkx
communities = partition.membership
node_communities = {node: membership for node, membership in zip(G.nodes(), communities)}

print("Node to community mapping:", node_communities)
