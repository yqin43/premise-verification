import json
import networkx as nx
import matplotlib.pyplot as plt

# Load JSON
with open('art_YN.json', 'r') as file:
    json_data = json.load(file)

G = nx.DiGraph()

cnt = 0
t_q = []
f_q = []
for data in json_data:
    # Add triple to graph
    G.add_edge(data["Ttriple"][0], data["Ttriple"][2], relation=data["Ttriple"][1])
    # t_q_ans.append([data["Ttriple"][0], data["Ttriple"][2], data["Ttriple"][1]])
    t_q.append(data["TPQ"])
    
    # Process paths
    for key, value in data.items():
        if key.startswith("path_"):
            if isinstance(value[0], list):
                for triple in value:
                    G.add_edge(triple[0], triple[2], relation=triple[1])
            else:
                G.add_edge(value[0], value[2], relation=value[1])
        if key.startswith("FPQ_"):
            f_q.append(data[key])

    cnt += 1

from torch_geometric.nn.nlp import SentenceTransformer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'sentence-transformers/all-roberta-large-v1'
model = SentenceTransformer(model_name)
model.eval()

# question_embs = model.encode(
#     t_q+f_q,
#     batch_size=256,
#     # output_device='cpu',
# )

import json
with open('KG-FPQ/logical_form.json', 'r') as file:
    lf = json.load(file)
import re

logical_form_lst = []
source_lst = []
target_lst = []
relationship_lst = []

for text in lf:
    # Extract Logical Form
    logical_form_match = re.search(r'Logical Form: (.+)', text)
    logical_form = logical_form_match.group(1).strip() if logical_form_match else None

    # Extract Source
    source_match = re.search(r'Source: (.+)', text)
    source = source_match.group(1).strip() if source_match else None

    # Extract Target
    target_match = re.search(r'Target: (.+)', text)
    target = target_match.group(1).strip() if target_match else None

    # Extract Relationship
    relationship_match = re.search(r'Relationship: (.+)', text)
    relationship = relationship_match.group(1).strip() if relationship_match else None

    logical_form = logical_form.strip('\'"')  # Remove leading/trailing ' or "
    source = source.strip('\'"')  # Remove leading/trailing ' or "
    target = target.strip('\'"')
    relationship = relationship.strip('\'"')


    logical_form_lst.append(logical_form)
    source_lst.append(source)
    target_lst.append(target)
    relationship_lst.append(relationship)

question_embs = model.encode(
    logical_form_lst,
    batch_size=256,
    # output_device='cpu',
)

# import g_retri
import retrieval

sub_g = []
for i in range(len(t_q+ f_q)):
    sub = retrieval.retrieval_via_pcst(G,question_embs[i])
    sub_g.append(sub)

import json
with open("g_retri_logical_form.json", "w") as f:
    json.dump(sub_g, f)
