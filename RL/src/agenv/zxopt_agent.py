from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
from torch.distributions.categorical import Categorical
from torch_geometric.nn import Sequential as geo_Sequential
import networkx as nx
from memory_profiler import profile



# torch.distributions.categoricalを継承して、
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, device="cpu"):
        if masks is None:
            masks = []
        self.masks = masks
        if len(self.masks) != 0:
            self.masks = masks.type(torch.BoolTensor).to(device)
            # torch.where(condition, true, false)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self, device):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)

class Agent(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_features(self, obs):
        pass

    @abstractmethod
    def action_fn(feat):
        pass

    def get_next_action(self, obs, device):
        feat = self.calculate_features(obs)
        return action_fn(feat)

# class ActorCriticAgent(Agent):
#     def __init__(self, actor, critic):
#         self.actor = actor
#         self.critic = critic
    
#     def calculate_features(self, obs):
        

#     def get_action_and_value(self, obs, device):
#         feat = self.calculate_features(obs)
#         action, logprob = self.actor(feat)
#         value = self.critic(feat)
#         return action, logprob, None, value, None, None



class AgentGNN(nn.Module):
    def __init__(
        self,
        envs,
        device,
        c_hidden=32,
        c_hidden_v=32,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        # self.obs_shape = envs.envs[0].shape
        # self.bin_required = int(np.ceil(np.log2(self.obs_shape)))
        self.envs = envs
        self.shape = 3000
        self.obs_shape = self.shape # NOTE(cgp): おそらく、policy_obsの長さの最大値
        #self.qubits = envs.envs[0].qubits

        c_in_p = 16
        c_in_v = 11
        edge_dim = 6
        edge_dim_v = 3
        self.global_attention_critic = geom_nn.GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(c_hidden, c_hidden),
                nn.ReLU(),
                nn.Linear(c_hidden, c_hidden),
                nn.ReLU(),
                nn.Linear(c_hidden, 1),
            ),
            nn=nn.Sequential(nn.Linear(c_hidden, c_hidden_v), nn.ReLU(), nn.Linear(c_hidden_v, c_hidden_v), nn.ReLU()),
        )

        self.critic_gnn = geo_Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    geom_nn.GATv2Conv(c_in_v, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim_v, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
            ],
        )

        self.actor_gnn = geom_nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    geom_nn.GATv2Conv(c_in_p, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (
                    geom_nn.GATv2Conv(c_hidden, c_hidden, edge_dim=edge_dim, add_self_loops=True),
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
                (nn.Linear(c_hidden, c_hidden),),
                nn.ReLU(),
                (nn.Linear(c_hidden, 1),),
            ],
        )

        self.critic_ff = nn.Sequential(
            nn.Linear(c_hidden_v, c_hidden_v),
            nn.ReLU(),
            nn.Linear(c_hidden_v, c_hidden_v),
            nn.ReLU(),
            nn.Linear(c_hidden_v, out_features=1),
        )

            # zxdiagramから(node_features, edge_index, edge_features、identifier)を作成して、torch.tensorで返す
    # node_features: [位相情報18個, 入力境界？, 出力境界？, フェーズガジェット？, 行動用ノードのための情報...]
    # edge_features: [普通のエッジ、lcomp, ident, pivot, fusion, stop]
    # identifier: 普通のノードは-1, 他はaction毎に特殊なスカラー
    def get_policy_feature_graph(self, graph, info) -> torch_geometric.data.Data:
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
                    node_feature[10] = 1.0

            # Extraction cost
            node_features.append(node_feature)

        # Relabel the nodes of the copied graph by adding n_nodes to each label
        n_nodes = len(p_graph.nodes())

        # Create tracking variable of label node to include new action nodes
        current_node = n_nodes
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
            identifier.append(node * self.shape + node)
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
            identifier.append(self.shape**2 + node)
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
            identifier.append(node1 * self.shape + node2)
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
        node_feature = [0 for _ in range(number_node_features)]
        node_feature[13] = 1.0
        node_features.append(node_feature)
        identifier.append(self.shape * (self.shape + 1) + 1)

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
        return torch_geometric.data.Data(
            x=x.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_features.to(self.device),
            y=identifier.to(self.device),
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
        
        return torch_geometric.data.Data(x=x_value.to(self.device), edge_index=edge_index_value.to(self.device), edge_attr=edge_features.to(self.device))


    def actor(self, x):
        logits = self.actor_gnn(x.x, x.edge_index, x.edge_attr)
        return logits

    def critic(self, x):
        features = self.critic_gnn(x.x, x.edge_index, x.edge_attr)
        aggregated = self.global_attention_critic(features, x.batch)
        return self.critic_ff(aggregated)

    # x: [Batch, Batch]
    # エージェントが次にとりたいと考えている行動を取得する. ついでに付加的な情報も返す. 
    def get_next_action(self, graph, info, action=None, device="cpu", testing=False):
        #  NOTE(cgp): vector envの場合、graphはリストになる. それぞれのgraphに対して特徴ベクトルを計算する.
        # この場合, policy_obsはtorch_geometric.data.Batchという形になる.
        # この Batch は、複数のグラフをまとめて扱うためのクラスで、これをself.actor()に入力することで、envs分のlogitsが得られる.
        # 下の方の、policy_obs.num_graphsのループで[policy_obs.batch == b]でバッチ毎に分割して処理する.
        if isinstance(graph,(tuple,list,np.ndarray)):
            policy_obs = torch_geometric.data.Batch.from_data_list([self.get_policy_feature_graph(g,i) for g, i in zip(graph,info)])
        else:
            policy_obs = self.get_policy_feature_graph(graph,info)
        logits = self.actor(policy_obs)
        
        # actionノードの
        batch_logits = torch.zeros([policy_obs.num_graphs, self.obs_shape]).to(device)
        act_mask = torch.zeros([policy_obs.num_graphs, self.obs_shape]).to(device)
        act_ids = torch.zeros([policy_obs.num_graphs, self.obs_shape]).to(device)
        action_logits = torch.tensor([]).to(device)

        for b in range(policy_obs.num_graphs):
            # get_policy_feature_graphのyにはidentifierが入っている
            ids = policy_obs.y[policy_obs.batch == b].to(device)
            # identifierが-1でないactionノードのみを取得
            action_nodes = torch.where(ids != -1)[0].to(device)
            # actionノードのnode_featureを取得
            probs = logits[policy_obs.batch == b][action_nodes]
            batch_logits[b, : probs.shape[0]] = probs.flatten()
            act_mask[b, : probs.shape[0]] = torch.tensor([True] * probs.shape[0])
            act_ids[b, : action_nodes.shape[0]] = ids[action_nodes]
            action_logits = torch.cat((action_logits, probs.flatten()), 0).reshape(-1)
            
        # Sample from each set of probs using Categorical
        categoricals = CategoricalMasked(logits=batch_logits, masks=act_mask, device=device)

        # Convert the list of samples back to a tensor
        # actionノードの
        if action is None:
            action = categoricals.sample()
            batch_id = torch.arange(policy_obs.num_graphs)
            action_id = act_ids[batch_id, action]

        else:
            action_id = torch.tensor([0]).to(device)
            
        if testing:
            return action.T, action_id.T
        
        logprob = categoricals.log_prob(action)
        entropy = categoricals.entropy(device)
        return action.T, logprob, entropy, torch.tensor(action_logits).to(device).reshape(-1, 1), action_id.T

    def get_value(self, graph):
        if isinstance(graph,(tuple,list)):
            value_obs = torch_geometric.data.Batch.from_data_list([self.get_value_feature_graph(g) for g in graph])
        else:
            value_obs = self.get_value_feature_graph(graph)
        values = self.critic(value_obs)
        values = values.squeeze(-1)
        return values# 