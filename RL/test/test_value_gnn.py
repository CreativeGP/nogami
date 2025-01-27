import os, sys, argparse, copy, random
from distutils.util import strtobool
from dataclasses import dataclass
from typing import List

import gym
import pandas as pd
import numpy as np
import pyzx as zx
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_geometric.nn as geom_nn
import networkx as nx
import matplotlib.pyplot as plt

# NOTE(cgp): あまりよくないらしいけど、ルートモジュールより上を経由するにはこうするしかないかも
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.util import rootdir, CustomizedSyncVectorEnv, for_minibatches, print_weight_summary, forward_hook, print_weights, print_grads, print_grad_summary, register_all_forward_hooks, register_all_backward_hooks, print_random_states, count_autograd_graph
from src.agenv.zxopt_agent import get_agent_from_state_dict
from src.agenv.zxopt_env import ZXEnvForTest

global device

def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_known_args()[0]

import json
import multiprocessing as mp
import argparse
args = parse_args()

class FeatureGraph():
    def __init__(self, config: List[str]):
        self.config = config
    # zxdiagramから(node_features, edge_index, edge_features、identifier)を作成して、torch.tensorで返す
    # node_features: [位相情報8個, 入力境界？, 出力境界？, フェーズガジェット？, 行動用ノードのための情報...]
    # edge_features: [普通のエッジ、lcomp, ident, pivot, fusion, stop]
    # identifier: 普通のノードは-1, 他はaction毎に特殊なスカラー
    def get_feature_graph(self, graph, info, mask_stop=False) -> torch_geometric.data.Data:
        """Enters the graph in format ZX"""
        piv_nodes = info['piv_nodes']
        lcomp_nodes = info['lcomp_nodes']
        iden_nodes = info['iden_nodes']
        graph_nx = nx.Graph()
        v_list = list(graph.vertices())  # vertices list
        e_list = list(graph.edge_set())  # edges list
        # create networkx graph
        graph_nx.add_nodes_from(v_list)
        graph_nx.add_edges_from(e_list)

        shape=3000

        # make the graph directed to duplicate edges
        graph_nx = graph_nx.to_directed()
        # relabel 0->N nodes
        mapping = {node: i for i, node in enumerate(graph_nx.nodes)}
        identifier = [0 for _ in mapping.items()]
        for key in mapping.keys():
            identifier[mapping[key]] = key
        p_graph = nx.relabel_nodes(graph_nx, mapping)

        # Features vector list

        neighbors_inputs = []
        for vertice in list(graph.inputs()):
            neighbors_inputs.append(list(graph.neighbors(vertice))[0])

        neighbors_outputs = []
        for vertice in list(graph.outputs()):
            neighbors_outputs.append(list(graph.neighbors(vertice))[0])

        node_features = []
        number_node_features = 16
        for node in sorted(p_graph.nodes):
            real_node = identifier[node]

            # Features: One-Hot phase, Frontier In, Frontier 0ut, Gadget, LC Node, PV Node,
            # STOP Node, ID Node, GadgetF, NOT INCLUDED Extraction Cost
            node_feature = [0.0 for _ in range(number_node_features)]

            # Frontier Node
            if real_node in graph.inputs():
                node_feature[8] = 1.0
            elif real_node in graph.outputs():
                node_feature[9] = 1.0
            else:
                # One-Hot phase
                oh_phase_idx = int(graph.phase(real_node) / (0.25))
                node_feature[oh_phase_idx] = 1.0

                if graph.neighbors(real_node) == 1:  # Phase Gadget
                    node_feature[10] = graph.neighbors(real_node)

            # Extraction cost
            node_features.append(node_feature)

        # Relabel the nodes of the copied graph by adding n_nodes to each label
        n_nodes = len(p_graph.nodes())

        # Create tracking variable of label node to include new action nodes
        current_node = n_nodes
        # NOTE(cgp): ここのedge_listの順番に決まりがないので、合わせることができない。順番が問題にならないならいいけど.
        edge_list = list(p_graph.edges)
        edge_features = []
        edge_feature_number = 6
        for edge in edge_list:
            # True: 1, False: 0. Features: Graph edge, NOT INCLUDED brings to frontier, NOT INCLUDED is brought by,
            # Removing Node-LC,Removing Node-PV, Removing Node-ID, Gadget fusion, Between Action
            node1, node2 = identifier[edge[0]], identifier[edge[1]]
            edge_feature = [0.0 for _ in range(edge_feature_number)]

            # Graph edge
            edge_feature[0] = 1.0
            edge_features.append(edge_feature)

        # アクションを表す特殊ノードを、関係ありそうなノードとエッジを張りながら追加
        # Add action nodes from lcomp and pivoting lists and connect them
        for node in lcomp_nodes:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[11] = 1.0
            node_features.append(node_feature)
            identifier.append(node * shape + node)
            # Connect the node to the rest of the graph
            graph_node = mapping[node]
            edge_list.append((mapping[node], current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[1] = 1.0
            edge_features.append(edge_feature)

            current_node += 1

        for node in iden_nodes:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[14] = 1.0
            node_features.append(node_feature)
            identifier.append(shape**2 + node)
            graph_node = mapping[node]
            edge_list.append((mapping[node], current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[3] = 1.0
            edge_features.append(edge_feature)

            current_node += 1

        for node1, node2 in piv_nodes:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[12] = 1.0
            node_features.append(node_feature)
            identifier.append(node1 * shape + node2)
            graph_node1 = mapping[node1]
            graph_node2 = mapping[node2]
            edge_list.append((graph_node1, current_node))
            edge_list.append((graph_node2, current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[2] = 1.0
            edge_features.append(edge_feature)
            edge_features.append(edge_feature)

            current_node += 1

        for idx, gadgetf in enumerate(info['gf_nodes']):

            node_feature = [0 for _ in range(number_node_features)]
            node_feature[15] = 1.0
            node_features.append(node_feature)
            identifier.append(-(idx + 2))

            for node in gadgetf:
                graph_node = mapping[node]
                edge_list.append((graph_node, current_node))
                edge_feature = [0 for _ in range(edge_feature_number)]
                edge_feature[4] = 1.0
                edge_features.append(edge_feature)

            current_node += 1

        # Add action for STOP node
        # STOPノードはほかのすべてのノードにエッジを張る
        if not mask_stop:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[13] = 1.0
            node_features.append(node_feature)
            identifier.append(shape * (shape + 1) + 1)

            for j in range(n_nodes, current_node):
                # Other actions feed Stop Node
                edge_list.append((j, current_node))
                edge_feature = [0 for _ in range(edge_feature_number)]
                edge_feature[5] = 1.0
                edge_features.append(edge_feature)

        # Create tensor objects
        x = torch.tensor(node_features).view(-1, number_node_features)
        x = x.type(torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        edge_features = torch.tensor(edge_features).view(-1, edge_feature_number)
        identifier[:n_nodes] = [-1] * n_nodes
        identifier = torch.tensor(identifier)

        # NOTE(cgp): x, yの長さはグラフのノード数によって異なる.
        # x: ノード特徴量
        # edge_index: グラフの接続関係を表す
        # edge_attr: エッジ特徴量
        # y: action identifier
        return torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=identifier,
        )
 


class GATv2(nn.Module):
    def __init__(self, args=None, ):
        super().__init__()
        # c_hidden=64#32
        # c_hidden_v=64#32
        c_hidden=32
        c_hidden_v=32

        c_in_p = 16
        c_in_v = 11
        edge_dim = 6
        edge_dim_v = 3

        self.critic_gnn = geom_nn.Sequential(
            "x, edge_index, edge_attr",
            [
                # (
                #     geom_nn.GCNConv(c_in_p, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                #     "x, edge_index -> x",
                # ),
                # nn.ReLU(),
                # (
                #     geom_nn.GCNConv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                #     "x, edge_index -> x",
                # ),
                # nn.ReLU(),
                # (
                #     geom_nn.GATv2Conv(c_in_v, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                #     "x, edge_index, edge_attr -> x",
                # ),
                # nn.ReLU(),
                # (
                #     geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                #     "x, edge_index, edge_attr -> x",
                # ),
                # nn.ReLU(),
                # (
                #     geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                #     "x, edge_index, edge_attr -> x",
                # ),
                # nn.ReLU(),
                # (
                #     geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                #     "x, edge_index, edge_attr -> x",
                # ),
                # nn.ReLU(),
                # (
                #     geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                #     "x, edge_index, edge_attr -> x",
                # ),
                # nn.ReLU(),
                (
                    geom_nn.GINConv(nn=nn.Sequential(
                        nn.Linear(c_in_p, c_hidden),
                        nn.ReLU(),
                        # nn.Linear(c_hidden, c_hidden),
                        # nn.ReLU(),
                    )),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GINConv(nn=nn.Sequential(
                        nn.Linear(c_hidden, c_hidden),
                        nn.ReLU(),
                        # nn.Linear(c_hidden, c_hidden),
                        # nn.ReLU(),
                    )),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),

            ],
        )


        self.global_attention_critic = geom_nn.aggr.AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(c_hidden, c_hidden),
                nn.ReLU(),
                nn.Linear(c_hidden, c_hidden),
                nn.ReLU(),
                nn.Linear(c_hidden, 1),
            ),
            nn=nn.Sequential(
                nn.Linear(c_hidden, c_hidden_v),
                nn.ReLU(),
                nn.Linear(c_hidden_v, c_hidden_v),
                nn.ReLU()
            ),
        )

        # self.critic_head_aggregation = geom_nn.aggr.SumAggregation()

        self.critic_ff = nn.Sequential(
            nn.Linear(c_hidden_v, c_hidden_v,),
            nn.ReLU(),
            nn.Linear(c_hidden_v, c_hidden_v),
            nn.ReLU(),
            nn.Linear(c_hidden_v, out_features=1),
        )

    def critic(self, x):
        features = self.critic_gnn(x.x, x.edge_index, x.edge_attr)
        aggregated = self.global_attention_critic(features, x.batch)
        return self.critic_ff(aggregated)

    # zxdiagramから(node_features, edge_index, edge_features、identifier)を作成して、torch.tensorで返す
    # node_features: [位相情報8個, 入力境界？, 出力境界？, フェーズガジェット？, 行動用ノードのための情報...]
    # edge_features: [普通のエッジ、lcomp, ident, pivot, fusion, stop]
    # identifier: 普通のノードは-1, 他はaction毎に特殊なスカラー
    def get_policy_feature_graph(self, graph, info, mask_stop=False) -> torch_geometric.data.Data:
        """Enters the graph in format ZX"""
        piv_nodes = info['piv_nodes']
        lcomp_nodes = info['lcomp_nodes']
        iden_nodes = info['iden_nodes']
        graph_nx = nx.Graph()
        v_list = list(graph.vertices())  # vertices list
        e_list = list(graph.edge_set())  # edges list
        # create networkx graph
        graph_nx.add_nodes_from(v_list)
        graph_nx.add_edges_from(e_list)

        shape=3000

        # make the graph directed to duplicate edges
        graph_nx = graph_nx.to_directed()
        # relabel 0->N nodes
        mapping = {node: i for i, node in enumerate(graph_nx.nodes)}
        identifier = [0 for _ in mapping.items()]
        for key in mapping.keys():
            identifier[mapping[key]] = key
        p_graph = nx.relabel_nodes(graph_nx, mapping)

        # Features vector list

        neighbors_inputs = []
        for vertice in list(graph.inputs()):
            neighbors_inputs.append(list(graph.neighbors(vertice))[0])

        neighbors_outputs = []
        for vertice in list(graph.outputs()):
            neighbors_outputs.append(list(graph.neighbors(vertice))[0])

        node_features = []
        number_node_features = 16
        for node in sorted(p_graph.nodes):
            real_node = identifier[node]

            # Features: One-Hot phase, Frontier In, Frontier 0ut, Gadget, LC Node, PV Node,
            # STOP Node, ID Node, GadgetF, NOT INCLUDED Extraction Cost
            node_feature = [0.0 for _ in range(number_node_features)]

            # Frontier Node
            if real_node in graph.inputs():
                node_feature[number_node_features-8+ 0] = 1.0
            elif real_node in graph.outputs():
                node_feature[number_node_features-8+ 1] = 1.0
            else:
                # One-Hot phase
                oh_phase_idx = int(graph.phase(real_node) / (0.25))
                node_feature[oh_phase_idx] = 1.0
                # node_feature[number_node_features-8+ 2] = len(graph.neighbors(real_node))

                # if graph.neighbors(real_node) == 1:  # Phase Gadget
                #     node_feature[2] = graph.neighbors(real_node)

            # Extraction cost
            node_features.append(node_feature)

        # Relabel the nodes of the copied graph by adding n_nodes to each label
        n_nodes = len(p_graph.nodes())

        # Create tracking variable of label node to include new action nodes
        current_node = n_nodes
        # NOTE(cgp): ここのedge_listの順番に決まりがないので、合わせることができない。順番が問題にならないならいいけど.
        edge_list = list(p_graph.edges)
        edge_features = []
        edge_feature_number = 6
        for edge in edge_list:
            # True: 1, False: 0. Features: Graph edge, NOT INCLUDED brings to frontier, NOT INCLUDED is brought by,
            # Removing Node-LC,Removing Node-PV, Removing Node-ID, Gadget fusion, Between Action
            node1, node2 = identifier[edge[0]], identifier[edge[1]]
            edge_feature = [0.0 for _ in range(edge_feature_number)]

            # Graph edge
            edge_feature[0] = 1.0
            edge_features.append(edge_feature)

        # アクションを表す特殊ノードを、関係ありそうなノードとエッジを張りながら追加
        # Add action nodes from lcomp and pivoting lists and connect them
        for node in lcomp_nodes:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[number_node_features-8+ 3] = 1.0
            node_features.append(node_feature)
            identifier.append(node * shape + node)
            # Connect the node to the rest of the graph
            graph_node = mapping[node]
            edge_list.append((mapping[node], current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[1] = 1.0
            edge_features.append(edge_feature)

            current_node += 1

        for node in iden_nodes:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[number_node_features-8+ 4] = 1.0
            node_features.append(node_feature)
            identifier.append(shape**2 + node)
            graph_node = mapping[node]
            edge_list.append((mapping[node], current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[3] = 1.0
            edge_features.append(edge_feature)

            current_node += 1

        for node1, node2 in piv_nodes:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[number_node_features-8+ 5] = 1.0
            node_features.append(node_feature)
            identifier.append(node1 * shape + node2)
            graph_node1 = mapping[node1]
            graph_node2 = mapping[node2]
            edge_list.append((graph_node1, current_node))
            edge_list.append((graph_node2, current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[2] = 1.0
            edge_features.append(edge_feature)
            edge_features.append(edge_feature)

            current_node += 1

        for idx, gadgetf in enumerate(info['gf_nodes']):

            node_feature = [0 for _ in range(number_node_features)]
            node_feature[number_node_features-8+ 6] = 1.0
            node_features.append(node_feature)
            identifier.append(-(idx + 2))

            for node in gadgetf:
                graph_node = mapping[node]
                edge_list.append((graph_node, current_node))
                edge_feature = [0 for _ in range(edge_feature_number)]
                edge_feature[4] = 1.0
                edge_features.append(edge_feature)

            current_node += 1

        # Add action for STOP node
        # STOPノードはほかのすべてのノードにエッジを張る
        if not mask_stop:
            node_feature = [0 for _ in range(number_node_features)]
            node_feature[number_node_features-8+ 7] = 1.0
            node_features.append(node_feature)
            identifier.append(shape * (shape + 1) + 1)

            for j in range(n_nodes, current_node):
                # Other actions feed Stop Node
                edge_list.append((j, current_node))
                edge_feature = [0 for _ in range(edge_feature_number)]
                edge_feature[5] = 1.0
                edge_features.append(edge_feature)
        # Create tensor objects
        x = torch.tensor(node_features).view(-1, number_node_features)
        x = x.type(torch.float32)
        edge_index = torch.tensor(edge_list).t().contiguous()
        edge_features = torch.tensor(edge_features).view(-1, edge_feature_number)
        identifier[:n_nodes] = [-1] * n_nodes
        identifier = torch.tensor(identifier)

        # NOTE(cgp): x, yの長さはグラフのノード数によって異なる.
        # x: ノード特徴量
        # edge_index: グラフの接続関係を表す
        # edge_attr: エッジ特徴量
        # y: action identifier
        return torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=identifier,
        )

    # critic用の観測情報を作る
    # zxdiagramから(node_features, edge_index, edge_features)を作成して、torch.tensorで返す
    # node_features: [位相情報8個, 入力境界？, 出力境界？, フェーズガジェット？]
    # edge_features: [1, 0, 0]固定
    def get_value_feature_graph(self, graph) -> torch_geometric.data.Data:
        Graph_nx = nx.Graph()
        v_list = list(graph.vertices())  # vertices list
        e_list = list(graph.edge_set())  # edges list
        # create networkx graph
        Graph_nx.add_nodes_from(v_list)
        Graph_nx.add_edges_from(e_list)
        
        # relabel 0->N nodes
        mapping = {node: i for i, node in enumerate(Graph_nx.nodes)}
        identifier = [0 for _ in mapping.items()]
        for key in mapping.keys():
            identifier[mapping[key]] = key
        V = nx.relabel_nodes(Graph_nx, mapping)

        neighbors_inputs = []
        for vertice in list(graph.inputs()):
            neighbors_inputs.append(list(graph.neighbors(vertice))[0])

        neighbors_outputs = []
        for vertice in list(graph.outputs()):
            neighbors_outputs.append(list(graph.neighbors(vertice))[0])

        node_features = []
        for node in sorted(V.nodes):
            real_node = identifier[node]
            # Features: Onehot PHASE, Frontier In, Frontier 0ut, Phase Gadget, NOT INCLUDED EXTRACTION COST
            node_feature = [0.0 for _ in range(11)]

            # Frontier Node
            if real_node in graph.inputs():
                node_feature[8] = 1.0
            elif real_node in graph.outputs():
                node_feature[9] = 1.0
            else:
                oh_phase_idx = int(graph.phase(real_node) / (0.25))
                node_feature[oh_phase_idx] = 1.0
                if graph.neighbors(real_node) == 1:  # Phase Gadget
                    node_feature[10] = 1.0

            node_features.append(node_feature)

        # Convert edges into bidirectional
        edge_list = list(V.edges)
        for node1, node2 in copy.copy(edge_list):
            edge_list.append((node2, node1))

        edge_features = []
        for node1, node2 in edge_list:
            # Edge in graph, pull node, pushed node.
            edge_feature = [1.0, 0.0, 0.0]
            edge_features.append(edge_feature)
        
        edge_index_value = torch.tensor(edge_list).t().contiguous()
        x_value = torch.tensor(node_features).view(-1, 11)
        x_value = x_value.type(torch.float32)
        edge_features = torch.tensor(edge_features).view(-1, 3)
        
        return torch_geometric.data.Data(
            x=x_value, 
            edge_index=edge_index_value, 
            edge_attr=edge_features
        )


@dataclass
class State:
    graph: zx.Graph
    nxfeat: torch_geometric.data.Data
    info: dict
    ret: float

# array: array-indexable array
def array_attr(arr, idx, attr):
    return np.array([getattr(a, attr) for a in arr[idx]])

def get_current_value_trace(states):
    values = []
    i = 0
    batching = 128
    while i < len(states):
        nxlist=[]
        for j in range(batching):
            if i+j >= len(states):
                break
            nxlist += [states[i+j].nxfeat]
            i += 1
        vs = model.critic(torch_geometric.data.Batch.from_data_list(nxlist)).detach()
        values += vs
    return values
def flat_grads(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters()])

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=UserWarning)

    seed = 8983440
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


    env = ZXEnvForTest(None, "gates")
    model = GATv2().to('cpu')
    print(model)

    # model.load_state_dict(torch.load("/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_zx-v0__b-seed-146__lavie__146__1731799751_8359936_model5x70_gates_new.pt"), strict=False)

    # N = 4096
    # for i in range(100):
    #     with open(f"/home/wsl/Research/nogami/zx/RL/training_ds/states_{i}.pkl", 'rb') as f:
    #         import pickle
    #         _states, _returns = pickle.load(f)
    #         for s,r in zip(_states,_returns):
    #             env.graph = s
    #             i = env.get_info()
    #             # states.append(State(s, model.get_policy_feature_graph(s,i), i, r))
    #             states.append(State(s, model.get_value_feature_graph(s), i, r))
    #         if len(states) > N:
    #             states = states[0:N]
    #             break
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    for i in range(13):
        states = []
        with open(f"/home/wsl/Research/nogami/zx/RL/training_ds/step_3.pkl", 'rb') as f:
            import pickle
            _states, _returns = pickle.load(f)
            for s,r in zip(_states,_returns):
                env.graph = s
                i = env.get_info()
                states.append(State(s, model.get_policy_feature_graph(s,i), i, r))
                # states.append(State(s, model.get_value_feature_graph(s), i, r))
            N=len(states)
        
        states = np.array(states)

        last_grad = None
        traj = []

        clip_coef = 0.1
        minibatch = 128
        for update in range(1):
            b_inds = np.arange(N)  
            for epoch in range(8):
                print("Epoch", epoch)

                # old_values = torch.Tensor(get_current_value_trace(states))
                np.random.shuffle(b_inds)  

                y_pred = np.array([])
                y_true = np.array([])
                for start in range(0, N, minibatch):
                    end = start + minibatch
                    mb_inds = b_inds[start:end]
                    mb_states = states[mb_inds]
                    mb_inputs = [s.nxfeat for s in mb_states]
                    mb_returns = torch.Tensor([s.ret for s in mb_states])

                    values = model.critic(torch_geometric.data.Batch.from_data_list(mb_inputs))
                    values = values.squeeze(-1)

                    clip_vloss=False
                    if clip_vloss:
                        v_loss_unclipped = (values - mb_returns) ** 2
                        v_clipped = old_values[mb_inds] + torch.clamp(
                            values - old_values[mb_inds],
                            -clip_coef,
                            clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((values-mb_returns) ** 2).mean()
                    # v_loss += 10000*(mb_returns.var()-values.var())**2
                    # v_val = values.var()
                    # v_ret = mb_returns.var()
                    # cov = (mb_returns*values).mean() - mb_returns.mean()*values.mean()
                    # v_loss = -10*cov / (torch.sqrt(v_val)*torch.sqrt(v_ret)) + (mb_returns.var()-values.var())**2
                    # v_loss = 10000 * torch.var(mb_returns - values) / torch.var(mb_returns)
                    optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip_coef)
                    # print()
                    # print_random_states(True)
                    optimizer.step()
                    # print(v_loss)
                    # print(count_autograd_graph(v_loss.grad_fn))
                    # print(values)
                    # print_grad_summary(model)

                    grad = flat_grads(model)
                    if last_grad is not None:
                        traj += [torch.dot(last_grad, grad)/torch.norm(last_grad)/torch.norm(grad)]
                    last_grad = grad.clone()

                    y_pred = np.concatenate((y_pred,values.detach()), axis=None)
                    y_true = np.concatenate((y_true,mb_returns), axis=None)
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                var_val = np.var(y_pred)
                var_ret = np.var(y_true)
                cov_pred_true = np.cov(y_pred, y_true)[0,1]

                print(y_true.mean(), y_pred.mean(), ((y_true - y_pred)**2).mean())

                print(f"expvar {explained_var:.3f}, v(val)={var_val:.3f}(σ={np.sqrt(var_val):.3f}), v(ret)={var_ret:.3f}(σ={np.sqrt(var_ret):.3f}),cov={cov_pred_true:.3f}")
                # x_min, x_max = plt.xlim()
                # # ylimをx軸の範囲と同じに設定
                # plt.ylim(x_min, x_max)
                # plt.scatter(y_true, y_pred)
                # plt.show()

            if update % 1 == 0:
                # 薄い色を重ねる。丸は小さく
                plt.scatter(y_true, y_pred, alpha=0.1, s=2)
                
                x_min, x_max = plt.xlim()
                # ylimをx軸の範囲と同じに設定
                plt.ylim(x_min, x_max)
                # plt.show()
            plt.plot(traj)
            # plt.show()
