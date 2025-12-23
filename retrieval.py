# g-retrieval
import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data


def retrieval_via_pcst(G, q_emb, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    G_edge_lst = list(G.edges(data=True))
    edge_attributes = [d['relation'] for _, _, d in G.edges(data=True)]
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    num_edges = len(edge_attributes)
    node_list = list(G.nodes())

    node_mapping = {n: i for i, n in enumerate(node_list)} # node:index
    edge_lst = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]

    from torch_geometric.nn.nlp import SentenceTransformer
    model_name = 'sentence-transformers/all-roberta-large-v1'
    model = SentenceTransformer(model_name)
    model.eval()
    x = model.encode(
        node_list,
        batch_size=256,
        # output_device='cpu',
    )
    edge_attr = model.encode(
        edge_attributes,
        batch_size=256,
        # output_device='cpu',
    )


    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, x)
        topk = min(topk, num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(edge_lst):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)
    
    # print([G_edge_lst[edge_i] for edge_i in selected_edges])
    loop_edge = [G_edge_lst[edge_i] for edge_i in selected_edges]
    to_return = []
    for u, v, data in loop_edge:
        relation = data.get("relation")  # Default to 'unknown' if not present
        to_return.append([u, relation, v])
    return to_return
    # debug
    # edge_index = edge_index[:, selected_edges]
    # selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    # n = textual_nodes.iloc[selected_nodes]
    # e = textual_edges.iloc[selected_edges]
    # desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    # mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    # x = x[selected_nodes]
    # edge_attr = edge_attr[selected_edges]
    # src = [mapping[i] for i in edge_index[0].tolist()]
    # dst = [mapping[i] for i in edge_index[1].tolist()]
    # edge_index = torch.LongTensor([src, dst])
    # print(edge_index, edge_attr)
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    #return data#