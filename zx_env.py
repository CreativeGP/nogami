import copy
import random
import signal
import time

from fractions import Fraction
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import gym
import networkx as nx
import torch
from torch_geometric.data import Batch, Data
import numpy as np
import pyzx as zx


from gym.spaces import Box, Discrete, Graph, MultiDiscrete
from pyzx.circuit import CNOT, HAD, SWAP, Circuit
from pyzx.extract import bi_adj, connectivity_from_biadj, greedy_reduction, id_simp, max_overlap, permutation_as_swaps

# from pyzx.gflow import gflow
from pyzx.graph.base import ET, VT, BaseGraph
from pyzx.linalg import Mat2
from pyzx.simplify import apply_rule, pivot
from pyzx.symbolic import Poly
from pyzx.utils import EdgeType, VertexType, toggle_edge


def handler(signum, frame):
    print("Teleport Reduce Fails")
    raise Exception("end of time")

class RLContext():
    def __init__(self):
        self.shape=3000
        self.device = "cuda"


    def update_state(self, graph):
        # TODO: これを書いて、stateを同じように更新できるようにしたのち
        # queue loopを書き直す.
        self.pivot_info_dict = self.match_pivot_parallel(graph) | self.match_pivot_boundary(graph) | self.match_pivot_gadget(graph)
        self.gadget_info_dict, self.gadgets = self.match_phase_gadgets(graph)
        self.gadget_fusion_ids = list(self.gadget_info_dict)

    def apply_rule(self, graph, edge_table, rem_vert, rem_edge, check_isolated_vertices):
        graph.add_edge_table(edge_table)
        graph.remove_edges(rem_edge)
        graph.remove_vertices(rem_vert)
        if check_isolated_vertices:
            graph.remove_isolated_vertices()
        return graph


    def policy_obs(self, graph):
        """Enters the graph in format ZX"""
        self.update_state(graph)
        piv_nodes = self.pivot_info_dict.keys()
        lcomp_nodes = self.match_lcomp(graph)
        iden_nodes = self.match_ids(graph)
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
        action_identifier = []
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
            action_identifier.append(node * self.shape + node)
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
            action_identifier.append(self.shape**2 + node)
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
            action_identifier.append(node1 * self.shape + node2)
            graph_node1 = mapping[node1]
            graph_node2 = mapping[node2]
            edge_list.append((graph_node1, current_node))
            edge_list.append((graph_node2, current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[2] = 1.0
            edge_features.append(edge_feature)
            edge_features.append(edge_feature)

            current_node += 1

        for idx, gadgetf in enumerate(self.gadget_fusion_ids):

            node_feature = [0 for _ in range(number_node_features)]
            node_feature[15] = 1.0
            node_features.append(node_feature)
            identifier.append(-(idx + 2))
            action_identifier.append(-(idx + 2))

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
        action_identifier.append(self.shape * (self.shape + 1) + 1)

        for j in range(n_nodes, current_node):
            # Other actions feed Stop Node
            edge_list.append((j, current_node))
            edge_feature = [0 for _ in range(edge_feature_number)]
            edge_feature[5] = 1.0
            edge_features.append(edge_feature)

        # Create tensor objects
        # x = torch.tensor(node_features).view(-1, number_node_features)
        # x = x.type(torch.float32)
        # edge_index = torch.tensor(edge_list).t().contiguous()
        # edge_features = torch.tensor(edge_features).view(-1, edge_feature_number)
        # identifier[:n_nodes] = [-1] * n_nodes
        # identifier = torch.tensor(identifier)
        return action_identifier

        # critic用の観測情報を作る
    # zxdiagramから(node_features, edge_index, edge_features)を作成して、torch.tensorで返す
    # node_features: [位相情報8個, 入力境界？, 出力境界？, フェーズガジェット？]
    # edge_features: [1, 0, 0]固定
    def value_obs(self, graph):
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
        return Data(x=x_value.to(self.device), edge_index=edge_index_value.to(self.device), edge_attr=edge_features.to(self.device))


    # NOTE(malick): これがコンテキストを変更しないように注意
    def step(self, graph, action):

        if int(action) == int(self.shape * (self.shape + 1) + 1):
            act_type = "STOP"
        elif int(action) > int(self.shape**2):
            act_type = "ID"
            #episode_stats["id"] +=1
            act_node1 = int(action) - int(self.shape**2)
        elif int(action) < 0:
            act_type = "GF"
            #episode_stats["gf"] += 1 
            act_node1 = self.gadget_fusion_ids[int(np.abs(action) - 2)] # Gadget fusion ids  start at 2
        else:
            act_node1, act_node2 = int(action // self.shape), int(action - action // self.shape * self.shape)
            if act_node1 == act_node2:
                act_type = "LC"
                #episode_stats["lc"] += 1
            else:
                if (act_node1, act_node2) in self.pivot_info_dict.keys() or (
                    act_node2,
                    act_node1,
                ) in self.pivot_info_dict.keys():
                    pv_type = self.pivot_info_dict[(act_node1, act_node2)][-1]
                    if pv_type == 0:
                        act_type = "PV"
                        # episode_stats["piv"] += 1
                    elif pv_type == 1:
                        act_type = "PVB"
                        # episode_stats["pivb"] +=1
                    else:
                        act_type = "PVG"
                        # episode_stats["pivg"] += 1
        # Update Stats
        # render_flag = 1
        # episode_len += 1
        # reward = 0
        # done = False

        if act_type == "LC":
            
            self.apply_rule(graph, *self.lcomp(graph, act_node1))
            action_id = 1
            node = [act_node1]
            
        elif act_type == "ID":
            
            neighbours = list(graph.neighbors(act_node1))
            types = graph.types()
            self.apply_rule(graph, *self.remove_ids(graph, act_node1))
            if types[neighbours[0]] != zx.VertexType.BOUNDARY and types[neighbours[1]] != zx.VertexType.BOUNDARY:
                self.apply_rule(graph, *self.spider_fusion(graph, neighbours))
            action_id = 3
            node = [act_node1]
            
        elif act_type == "PV" or act_type == "PVG" or act_type == "PVB":
            
            pv_type = self.pivot_info_dict[(act_node1, act_node2)][-1]
            if pv_type == 0:
                self.apply_rule(graph, *self.pivot(graph, act_node1, act_node2))
            else:
                self.apply_rule(graph, *self.pivot_gadget(graph, act_node1, act_node2))
            action_id = 2
            node = [act_node1, act_node2]
            
        elif act_type == "GF":
            self.apply_rule(graph, *self.merge_phase_gadgets(graph, act_node1))
            action_id = 6
            node = act_node1
            
        elif act_type == "STOP":
            action_id = 0
            node = [-1]
            
        else:
            action_id = 5  
            reward = 0.0
            node = [-1]
            
        
        graph = graph.copy() #Relabel nodes due to PYZX not keeping track of node id properly.
        graph = graph.copy()
        graph.normalize()

        return graph, act_type
        
        try:
            circuit = zx.extract_circuit(graph, up_to_perm=True)
            circuit = circuit.to_basic_gates()
            circ = zx.basic_optimization(circuit).to_basic_gates()
            circuit_data = get_data(circ)
            new_gates = circuit_data[gate_type]
        except:
            new_gates = np.inf
            act_type = "STOP"
        
        action_pattern.append([act_type, new_gates-current_gates])
        reward = 0
        if new_gates < min_gates:
            min_gates = new_gates
            final_circuit = circ            
            
        if new_gates <= min_gates:
            opt_episode_len = episode_len
            best_action_stats = copy.deepcopy(episode_stats)

        reward += (current_gates - new_gates) / max_compression
        episode_reward += reward

        pivot_info_dict = match_pivot_parallel() | match_pivot_boundary() | match_pivot_gadget()
        gadget_info_dict, gadgets = match_phase_gadgets()
        gadget_fusion_ids = list(gadget_info_dict)
        # Obtain Length of Remaining Actions:
        remaining_pivot = len(pivot_info_dict.keys())
        remaining_lcomp = len(match_lcomp())
        remaining_ids = len(match_ids())
        remaining_gadget_fusions = len(gadget_fusion_ids)
        remaining_actions = remaining_pivot + remaining_lcomp + remaining_ids + remaining_gadget_fusions


        # End episode if there are no remaining actions or Maximum Length Reached or Incorrect Action Selected
        if (
            remaining_actions == 0 or act_type == "STOP" or episode_len == max_episode_len
        ):  
            
            reward += (min(pyzx_gates, basic_opt_data[gate_type], initial_stats[gate_type])-new_gates)/max_compression
            
            if min_gates < min(pyzx_gates, basic_opt_data[gate_type], initial_stats[gate_type]):
                win_vs_pyzx = 1
            elif min_gates == min(pyzx_gates, basic_opt_data[gate_type], initial_stats[gate_type]):
                win_vs_pyzx = 0
            else:
                win_vs_pyzx = -1
            
            done = True

            print("Win vs Pyzx: ", win_vs_pyzx, " Episode Gates: ", min_gates, "Cflow_gates: ", pyzx_gates, "Episode Len", episode_len, "Opt Episode Len", opt_episode_len)
            return (
                graph,
                reward,
                done,
                False,
                {
                    "action": action_id,
                    "remaining_lcomp_size": remaining_lcomp,
                    "remaining_pivot_size": remaining_pivot,
                    "remaining_id_size": remaining_ids,
                    "max_reward_difference": max_reward,
                    "action_pattern": action_pattern,
                    "opt_episode_len": opt_episode_len - episode_len,
                    "episode_len": episode_len,
                    "pyzx_stats": pyzx_data,
                    "rl_stats": get_data(final_circuit),
                    "no_opt_stats": no_opt_stats,
                    "swap_cost": swap_cost,
                    "pyzx_swap_cost": pyzx_swap_cost,
                    "pyzx_gates": pyzx_gates,
                    "rl_gates": get_data(final_circuit)[gate_type],
                    "bo_stats": basic_opt_data,
                    "initial_stats": initial_stats,
                    "win_vs_pyzx": win_vs_pyzx,
                    "min_gates": min_gates,
                    "graph_obs": [policy_obs(), value_obs()],
                    "final_circuit": final_circuit,
                    "action_stats": [best_action_stats["pivb"], 
                                        best_action_stats["pivg"],
                                        best_action_stats["piv"],
                                        best_action_stats["lc"],
                                        best_action_stats["id"],
                                        best_action_stats["gf"]],
                    "depth": final_circuit.depth(),
                    "initial_depth": initial_depth
                },
            )

        current_gates = new_gates

        return (
            graph,
            reward,
            done,
            False,
            {
                "action": action_id,
                "nodes": node,
                "graph_obs": [policy_obs(), value_obs()],
            },
        )

    MatchLcompType = Tuple[VT,Tuple[VT,...]]
    def match_lcomp(
            self,
        graph,
        vertexf: Optional[Callable[[VT],bool]] = None, 
        num: int = -1, 
        check_edge_types: bool = True,
        allow_interacting_matches: bool = False
        ) -> List[MatchLcompType[VT]]:
        """Finds matches of the local complementation rule.
        
        :param g: An instance of a ZX-graph.
        :param num: Maximal amount of matchings to find. If -1 (the default)
        tries to find as many as possible.
        :param check_edge_types: Whether the method has to check if all the edges involved
        are of the correct type (Hadamard edges).
        :param vertexf: An optional filtering function for candidate vertices, should
        return True if a vertex should be considered as a match. Passing None will
        consider all vertices.
        :param allow_interacting_matches: Whether or not to allow matches which overlap,
            hence can not all be applied at once. Defaults to False.
        :rtype: List of 2-tuples ``(vertex, neighbors)``.
        """
        if vertexf is not None: candidates = set([v for v in graph.vertices() if vertexf(v)])
        else: candidates = graph.vertex_set()
        
        phases = graph.phases()
        types = graph.types()
        
        i = 0
        m: List[MatchLcompType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            v = candidates.pop()
            
            if types[v] != VertexType.Z: continue
            if phases[v] not in (Fraction(1,2), Fraction(3,2)): continue
            if graph.is_ground(v): continue

            if check_edge_types and not (
                all(graph.edge_type(e) == EdgeType.HADAMARD for e in graph.incident_edges(v))
                ): continue

            vn = list(graph.neighbors(v))
            if any(types[n] != VertexType.Z for n in vn): continue
            
            #m.append((v,tuple(vn)))
            if len(graph.neighbors(v)) ==1:  #Phase gadget of pi/2 can not be selected
                continue
            flag = False
            for neigh_pg in graph.neighbors(v): #If root node of phase gadget is a neighbor of candidate node, node can not be selected.
                for neigh_pg2 in graph.neighbors(neigh_pg):
                    if len(graph.neighbors(neigh_pg2))==1:
                        flag = True
            if flag:
                continue
            m.append(v)
            i += 1
            
            if allow_interacting_matches: continue
            for n in vn: candidates.discard(n)
        return m

    RewriteOutputType = Tuple[Dict[ET, List[int]], List[VT], List[ET], bool]
    MatchPivotType = Tuple[VT, VT, Tuple[VT, ...], Tuple[VT, ...]]

    def match_pivot_parallel(
            self,
            graph,
        matchf: Optional[Callable[[ET], bool]] = None,
        num: int = -1,
        check_edge_types: bool = True,
        allow_interacting_matches: bool = False,
    ) -> List[MatchPivotType[VT]]:
        """Finds matches of the pivot rule.

        :param g: An instance of a ZX-graph.
        :param num: Maximal amount of matchings to find. If -1 (the default)
        tries to find as many as possible.
        :param check_edge_types: Whether the method has to check if all the edges involved
        are of the correct type (Hadamard edges).
        :param matchf: An optional filtering function for candidate edge, should
        return True if a edge should considered as a match. Passing None will
        consider all edges.
        :param allow_interacting_matches: Whether or not to allow matches which overlap,
            hence can not all be applied at once. Defaults to False.
        :rtype: List of 4-tuples. See :func:`pivot` for the details.
        """
        if matchf is not None:
            candidates = set([e for e in graph.edges() if matchf(e)])
        else:
            candidates = graph.edge_set()

        types = graph.types()
        phases = graph.phases()
        matches_dict = {}
        i = 0
        m: List[MatchPivotType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            e = candidates.pop()
            if check_edge_types and graph.edge_type(e) != EdgeType.HADAMARD:
                continue

            v0, v1 = graph.edge_st(e)
            if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z):
                continue
            if any(phases[v] not in (0, 1) for v in (v0, v1)):
                continue
            if graph.is_ground(v0) or graph.is_ground(v1):
                continue

            invalid_edge = False
            v0n = list(graph.neighbors(v0))
            v0b = []
            for n in v0n:
                if types[n] == VertexType.Z and graph.edge_type(graph.edge(v0, n)) == EdgeType.HADAMARD:
                    pass
                elif types[n] == VertexType.BOUNDARY:
                    v0b.append(n)
                else:
                    invalid_edge = True
                    break
            if invalid_edge:
                continue

            v1n = list(graph.neighbors(v1))
            v1b = []
            for n in v1n:
                if types[n] == VertexType.Z and graph.edge_type(graph.edge(v1, n)) == EdgeType.HADAMARD:
                    pass
                elif types[n] == VertexType.BOUNDARY:
                    v1b.append(n)
                else:
                    invalid_edge = True
                    break
            if invalid_edge:
                continue
            if len(v0b) + len(v1b) > 1:
                continue

            m.append((v0, v1, tuple(v0b), tuple(v1b)))
            matches_dict[(v0, v1)] = (tuple(v0b), tuple(v1b), 0)
            i += 1

        return matches_dict

    def match_pivot_gadget(self,graph,
        matchf: Optional[Callable[[ET], bool]] = None, num: int = -1, allow_interacting_matches: bool = False
    ) -> List[MatchPivotType[VT]]:
        """Like :func:`match_pivot_parallel`, but except for pairings of
        Pauli vertices, it looks for a pair of an interior Pauli vertex and an
        interior non-Clifford vertex in order to gadgetize the non-Clifford vertex."""
        if matchf is not None:
            candidates = set([e for e in graph.edges() if matchf(e)])
        else:
            candidates = graph.edge_set()

        types = graph.types()
        phases = graph.phases()
        matches_dict = {}
        i = 0
        m: List[MatchPivotType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            e = candidates.pop()
            v0, v1 = graph.edge_st(e)
            if not all(types[v] == VertexType.Z for v in (v0, v1)):
                continue

            if phases[v0] not in (0, 1):
                if phases[v1] in (0, 1):
                    v0, v1 = v1, v0
                else:
                    continue
            elif phases[v1] in (0, 1):
                continue  # Now v0 has a Pauli phase and v1 has a non-Pauli phase

            if graph.is_ground(v0):
                continue

            v0n = list(graph.neighbors(v0))
            v1n = list(graph.neighbors(v1))
            if len(v1n) == 1:
                continue  # It is a phase gadget
            if any(types[n] != VertexType.Z for vn in (v0n, v1n) for n in vn):
                continue

            bad_match = False
            edges_to_discard = []
            for i, neighbors in enumerate((v0n, v1n)):
                for n in neighbors:
                    if types[n] != VertexType.Z:
                        bad_match = True
                        break
                    ne = list(graph.incident_edges(n))
                    if i == 0 and len(ne) == 1 and not (e == ne[0]):  # v0 is a phase gadget
                        bad_match = True
                        break
                    edges_to_discard.extend(ne)
                if bad_match:
                    break
            if bad_match:
                continue

            m.append((v0, v1, tuple(), tuple()))
            matches_dict[(v0, v1)] = (tuple(), tuple(), 2)
            i += 1

        return matches_dict

    def match_pivot_boundary(self, graph,
        matchf: Optional[Callable[[VT], bool]] = None, num: int = -1, allow_interacting_matches: bool = False
    ) -> List[MatchPivotType[VT]]:
        """Like :func:`match_pivot_parallel`, but except for pairings of
        Pauli vertices, it looks for a pair of an interior Pauli vertex and a
        boundary non-Pauli vertex in order to gadgetize the non-Pauli vertex."""
        if matchf is not None:
            candidates = set([v for v in graph.vertices() if matchf(v)])
        else:
            candidates = graph.vertex_set()

        phases = graph.phases()
        types = graph.types()
        matches_dict = {}
        i = 0
        consumed_vertices: Set[VT] = set()
        m: List[MatchPivotType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            v = candidates.pop()
            if types[v] != VertexType.Z or phases[v] not in (0, 1) or graph.is_ground(v):
                continue

            good_vert = True
            w = None
            bound = None
            for n in graph.neighbors(v):
                if (
                    types[n] != VertexType.Z
                    or len(graph.neighbors(n)) == 1
                    or n in consumed_vertices
                    or graph.is_ground(n)
                ):
                    good_vert = False
                    break

                boundaries = []
                wrong_match = False
                for b in graph.neighbors(n):
                    if types[b] == VertexType.BOUNDARY:
                        boundaries.append(b)
                    elif types[b] != VertexType.Z:
                        wrong_match = True
                if len(boundaries) != 1 or wrong_match:
                    continue  # n is not on the boundary or has too many boundaries or has neighbors of wrong type
                if phases[n] and hasattr(phases[n], "denominator") and phases[n].denominator == 2:
                    w = n
                    bound = boundaries[0]
                if not w:
                    w = n
                    bound = boundaries[0]
            if not good_vert or w is None:
                continue
            assert bound is not None

            m.append((v, w, tuple(), tuple([bound])))
            matches_dict[(v, w)] = (tuple(), tuple([bound]), 1)
            i += 1
        return matches_dict

    def lcomp(self, graph, node):
        phase = graph.phase(node)
        neighbors = list(graph.neighbors(node))
        edge_table = dict()
        vertice = []
        vertice.append(node)
        n = len(neighbors)
        if phase.numerator == 1:
            graph.scalar.add_phase(Fraction(1, 4))
        else:
            graph.scalar.add_phase(Fraction(7, 4))
        graph.scalar.add_power((n - 2) * (n - 1) // 2)
        for i in range(n):
            graph.add_to_phase(neighbors[i], -phase)
            for j in range(i + 1, n):
                edge_neigh = graph.edge(neighbors[i], neighbors[j])  # edge type between neighbours
                he = edge_table.get(edge_neigh, [0, 0])[1]
                edge_table[edge_neigh] = [0, he + 1]

        return (edge_table, vertice, [], True)

    def pivot(self, graph, v0, v1) -> RewriteOutputType[ET, VT]:
        """Perform a pivoting rewrite, given a list of matches as returned by
        ``match_pivot(_parallel)``. A match is itself a list where:

        ``m[0]`` : first vertex in pivot.
        ``m[1]`` : second vertex in pivot.
        ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
        ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
        """
        rem_verts: List[VT] = []
        rem_edges: List[ET] = []
        etab: Dict[ET, List[int]] = dict()
        m = [0, 0, 0, 0, 0]

        m[0], m[1] = v0, v1
        # (v0b, v1b, 0)
        m[2], m[3], _ = self.pivot_info_dict[(v0, v1)]
        phases = graph.phases()

        n = [set(graph.neighbors(m[0])), set(graph.neighbors(m[1]))]
        for i in range(2):
            n[i].remove(m[1 - i])  # type: ignore # Really complex typing situation
            if len(m[i + 2]) == 1:
                n[i].remove(m[i + 2][0])  # type: ignore

        n.append(n[0] & n[1])  #  n[2] <- non-boundary neighbors of m[0] and m[1]
        n[0] = n[0] - n[2]  #  n[0] <- non-boundary neighbors of m[0] only
        n[1] = n[1] - n[2]  #  n[1] <- non-boundary neighbors of m[1] only

        es = (
            [graph.edge(s, t) for s in n[0] for t in n[1]]
            + [graph.edge(s, t) for s in n[1] for t in n[2]]
            + [graph.edge(s, t) for s in n[0] for t in n[2]]
        )
        k0, k1, k2 = len(n[0]), len(n[1]), len(n[2])
        graph.scalar.add_power(k0 * k2 + k1 * k2 + k0 * k1)

        for v in n[2]:
            if not graph.is_ground(v):
                graph.add_to_phase(v, 1)

        if phases[m[0]] and phases[m[1]]:
            graph.scalar.add_phase(Fraction(1))
        if not m[2] and not m[3]:
            graph.scalar.add_power(-(k0 + k1 + 2 * k2 - 1))
        elif not m[2]:
            graph.scalar.add_power(-(k1 + k2))
        else:
            graph.scalar.add_power(-(k0 + k2))

        for i in range(2):  # if m[i] has a phase, it will get copied on to the neighbors of m[1-i]:
            a = phases[m[i]]  # type: ignore
            if a:
                for v in n[1 - i]:
                    if not graph.is_ground(v):
                        graph.add_to_phase(v, a)
                for v in n[2]:
                    if not graph.is_ground(v):
                        graph.add_to_phase(v, a)

            if not m[i + 2]:
                rem_verts.append(m[1 - i])  # type: ignore # if there is no boundary, the other vertex is destroyed
            else:
                e = graph.edge(m[i], m[i + 2][0])  # type: ignore # if there is a boundary, toggle whether it is an h-edge or a normal edge
                new_e = graph.edge(m[1 - i], m[i + 2][0])  # type: ignore # and point it at the other vertex
                ne, nhe = etab.get(new_e, [0, 0])
                if graph.edge_type(e) == EdgeType.SIMPLE:
                    nhe += 1
                elif graph.edge_type(e) == EdgeType.HADAMARD:
                    ne += 1
                etab[new_e] = [ne, nhe]
                rem_edges.append(e)

        for e in es:
            nhe = etab.get(e, (0, 0))[1]
            etab[e] = [0, nhe + 1]

        return (etab, rem_verts, rem_edges, True)

    def pivot_gadget(self, graph, v0, v1) -> RewriteOutputType[ET, VT]:
        """Performs the gadgetizations required before applying pivots.
        ``m[0]`` : interior pauli vertex
        ``m[1]`` : interior non-pauli vertex to gadgetize
        ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
        ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
        """
        vertices_to_gadgetize = v1
        self.gadgetize(graph, vertices_to_gadgetize)
        return self.pivot(graph, v0, v1)

    def gadgetize(self, graph, vertices: VT) -> None:
        """Helper function which pulls out a list of vertices into gadgets"""
        edge_list = []

        inputs = graph.inputs()
        phases = graph.phases()
        v = vertices
        if any(n in inputs for n in graph.neighbors(v)):
            mod = 0.5
        else:
            mod = -0.5

        vp = graph.add_vertex(VertexType.Z, -2, graph.row(v) + mod, phases[v])
        v0 = graph.add_vertex(VertexType.Z, -1, graph.row(v) + mod, 0)
        graph.set_phase(v, 0)

        edge_list.append(graph.edge(v, v0))
        edge_list.append(graph.edge(v0, vp))

        # get python attributes

        # if graph.phase_tracking:
        #     graph.unfuse_vertex(vp, v)

        graph.add_edges(edge_list, EdgeType.HADAMARD)
        return

    MatchGadgetType = Tuple[VT, int, List[VT], Dict[VT, VT]]

    def match_phase_gadgets(self, graph, vertexf: Optional[Callable[[VT], bool]] = None) -> List[MatchGadgetType[VT]]:
        """Determines which phase gadgets act on the same vertices, so that they can be fused together.

        :param g: An instance of a ZX-graph.
        :rtype: List of 4-tuples ``(leaf, parity_length, other axels with same targets, leaf dictionary)``.
        1.leaf is a vertex that represents a phase gadget
        2.parity_length is the number of vertices that the phase gadget acts on
        3.other_axels is a list of other phase gadgets that act on the same vertices as leaf
        4.leaf_dict is a dictionary that maps each phase gadget to its corresponding phase node
        """
        if vertexf is not None:
            candidates = set([v for v in graph.vertices() if vertexf(v)])
        else:
            candidates = graph.vertex_set()
        gadget_info_dict = {}
        phases = graph.phases()

        parities: Dict[FrozenSet[VT], List[VT]] = dict()
        gadgets: Dict[VT, VT] = dict()
        inputs = graph.inputs()
        outputs = graph.outputs()
        # First we find all the phase-gadgets, and the list of vertices they act on
        for v in candidates:
            non_clifford = phases[v] != 0 and getattr(phases[v], "denominator", 1) > 2
            if isinstance(phases[v], Poly):
                non_clifford = True
            if non_clifford and len(list(graph.neighbors(v))) == 1:
                n = list(graph.neighbors(v))[0]
                if phases[n] not in (0, 1):
                    continue  # Not a real phase gadget (happens for scalar diagrams)
                if n in gadgets:
                    continue  # Not a real phase gadget (happens for scalar diagrams)
                if n in inputs or n in outputs:
                    continue  # Not a real phase gadget (happens for non-unitary diagrams)
                gadgets[n] = v
                par = frozenset(set(graph.neighbors(n)).difference({v}))
                if par in parities:
                    parities[par].append(n)
                else:
                    parities[par] = [n]

        for par, gad in parities.items():
            if len(gad) == 1:
                n = gad[0]
                if phases[n] != 0:
                    continue
            else:
                # n = gad.pop()
                gadget_info_dict[tuple(gad)] = len(par)

        return gadget_info_dict, gadgets

    def merge_phase_gadgets(self, graph, vertexs: Tuple[VT]) -> RewriteOutputType[ET, VT]:
        """v0,v1"""
        """Given the output of :func:``match_phase_gadgets``, removes phase gadgets that act on the same set of targets."""
        rem = []
        phases = graph.phases()
        par_num = self.gadget_info_dict[vertexs]
        n = vertexs[0]
        gad = list(vertexs[1:])
        #gadgets = gadgets

        v = self.gadgets[n]
        if len(gad) == 0:
            if phases[n] != 0:
                graph.scalar.add_phase(phases[v])
                if graph.phase_tracking:
                    graph.phase_negate(v)
                phase = -phases[v]
        else:
            phase = sum((1 if phases[w] == 0 else -1) * phases[self.gadgets[w]] for w in gad + [n]) % 2
            for w in gad + [n]:
                if phases[w] != 0:
                    graph.scalar.add_phase(phases[self.gadgets[w]])
                    if graph.phase_tracking:
                        graph.phase_negate(self.gadgets[w])
            graph.scalar.add_power(-((par_num - 1) * len(gad)))
        graph.set_phase(v, phase)
        graph.set_phase(n, 0)
        othertargets = [self.gadgets[w] for w in gad]
        rem.extend(gad)
        rem.extend(othertargets)
        for w in othertargets:
            if graph.phase_tracking:
                graph.fuse_phases(v, w)
            if graph.merge_vdata is not None:
                graph.merge_vdata(v, w)
        return ({}, rem, [], False)

    def spider_fusion(self, graph, neighs):
        rem_verts = []
        etab = dict()

        if graph.row(neighs[0]) == 0:
            v0, v1 = neighs[1], neighs[0]
        else:
            v0, v1 = neighs[0], neighs[1]
        ground = graph.is_ground(v0) or graph.is_ground(v1)
        if ground:
            graph.set_phase(v0, 0)
            graph.set_ground(v0)
        else:
            graph.add_to_phase(v0, graph.phase(v1))
        if graph.phase_tracking:
            graph.fuse_phases(v0, v1)
        # always delete the second vertex in the match
        rem_verts.append(v1)
        # edges from the second vertex are transferred to the first
        for w in graph.neighbors(v1):
            if v0 == w:
                continue
            e = graph.edge(v0, w)
            if e not in etab:
                etab[e] = [0, 0]
            etab[e][graph.edge_type(graph.edge(v1, w)) - 1] += 1
        return (etab, rem_verts, [], True)

    def remove_ids(self, graph, node):
        neigh = graph.neighbors(node)
        v0, v1 = neigh
        if graph.edge_type(graph.edge(node, v0)) != graph.edge_type(
            graph.edge(node, v1)
        ):  # exactly one of them is a hadamard edge
            et = zx.EdgeType.HADAMARD
        else:
            et = zx.EdgeType.SIMPLE
        # create dict, rem_vertexs
        etab = dict()
        e = graph.edge(v0, v1)
        if not e in etab:
            etab[e] = [0, 0]
        if et == zx.EdgeType.SIMPLE:
            etab[e][0] += 1
        else:
            etab[e][1] += 1
        return (etab, [node], [], False)

    def match_ids(self, graph):
        candidates = graph.vertex_set()
        types = graph.types()
        phases = graph.phases()
        m = []
        while len(candidates) > 0:
            v = candidates.pop()
            if phases[v] != 0 or not zx.utils.vertex_is_zx(types[v]) or graph.is_ground(v):
                continue
            neigh = graph.neighbors(v)
            if len(neigh) != 2:
                continue
            v0, v1 = neigh
            if (
                graph.is_ground(v0)
                and types[v1] == zx.VertexType.BOUNDARY
                or graph.is_ground(v1)
                and types[v0] == zx.VertexType.BOUNDARY
            ):
                # Do not put ground spiders on the boundary
                continue
            m.append(v)
        return m

    def get_info(self, graph):
        return {
            "graph_obs": graph,
            # "full_reduce_time": full_reduce_end-full_reduce_start,
            "piv_nodes": self.pivot_info_dict,
            "lcomp_nodes": self.match_lcomp(graph),
            "iden_nodes": self.match_ids(graph),
            "gf_nodes": self.gadget_info_dict,
        }