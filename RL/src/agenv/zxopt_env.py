import copy, random, signal, time, argparse

from fractions import Fraction
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import gym
import networkx as nx
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

from src.util import rootdir, print_random_states, ActionHistory

def handler(signum, frame):
    print("Teleport Reduce Fails")
    raise Exception("end of time")

class ZXEnvBase(gym.Env):
    def __init__(
            self,
            silent: bool = False,
            args: argparse.Namespace = None):
        # self.device = "cuda"
        self.silent = silent
        self.args = args
        self.clifford = False
        self.shape = 3000
        # d["qubits"] = circuit.qubits
        # d["gates"] = total
        # d["tcount"] = tcount
        # d["clifford"] = clifford
        # d["CNOT"] = cnots
        # d["CZ"] = cz
        # d["had"] = hadamards
        # d["twoqubits"] = twoqubits
        # self.gate_type = "gates"
        self.episode_len = 0
        self.max_episode_len = 75
        self.cumulative_reward_episodes = 0
        self.win_episodes = 0
        # NOTE(cgp): なんで55になってるんだろう
        self.max_compression = 20
        # self.max_compression = 55

        # Unused variables but required for gym
        self.action_space = Discrete(1)
        self.single_action_space = Discrete(1)
        self.observation_space = Graph(node_space=Box(low=-1, high=1, shape=(3,)), edge_space=Discrete(3), seed=42)
        self.single_observation_space = Graph(
            node_space=Box(low=-1, high=1, shape=(17,)), edge_space=Discrete(3), seed=42
        )


    def step(self, action):
        if int(action) == int(self.shape) * (int(self.shape) + 1) + 1:
            act_type = "STOP"
        elif int(action) > int(self.shape)**2:
            act_type = "ID"
            self.episode_stats["id"] +=1
            act_node1 = int(action) - int(self.shape)**2
        elif int(action) < 0:
            act_type = "GF"
            self.episode_stats["gf"] += 1 
            act_node1 = self.gadget_fusion_ids[int(np.abs(action) - 2)] # Gadget fusion ids  start at 2
        else:
            act_node1, act_node2 = int(action // self.shape), int(action - action // self.shape * self.shape)
            if act_node1 == act_node2:
                act_type = "LC"
                self.episode_stats["lc"] += 1
            else:
                if (act_node1, act_node2) in self.pivot_info_dict.keys() or (
                    act_node2,
                    act_node1,
                ) in self.pivot_info_dict.keys():
                    pv_type = self.pivot_info_dict[(act_node1, act_node2)][-1]
                    if pv_type == 0:
                        act_type = "PV"
                        self.episode_stats["piv"] += 1
                    elif pv_type == 1:
                        act_type = "PVB"
                        self.episode_stats["pivb"] +=1
                    else:
                        act_type = "PVG"
                        self.episode_stats["pivg"] += 1
        
        vs = []
        if 'act_node1' in locals():
            vs.append(act_node1)
        if 'act_node2' in locals():
            vs.append(act_node2)
        return self.manual_step(act_type, vs)



    # self.graphの扱いがひどいので外部から操作する際は注意
    def manual_step(self, act_type, vs):
        if len(vs) == 1:
            act_node1 = vs[0]
        elif len(vs) == 2:
            act_node1 = vs[0]
            act_node2 = vs[1]


        # Update Stats
        self.render_flag = 1
        self.episode_len += 1
        reward = 0
        done = False
        

        if act_type == "LC":
            
            self.apply_rule(*self.lcomp(act_node1))
            action_id = 1
            node = [act_node1]
            
        elif act_type == "ID":
            
            neighbours = list(self.graph.neighbors(act_node1))
            types = self.graph.types()
            self.apply_rule(*self.remove_ids(act_node1))
            if types[neighbours[0]] != zx.VertexType.BOUNDARY and types[neighbours[1]] != zx.VertexType.BOUNDARY:
                self.apply_rule(*self.spider_fusion(neighbours))
            action_id = 3
            node = [act_node1]
            
        elif act_type == "PV" or act_type == "PVG" or act_type == "PVB":
            
            pv_type = self.pivot_info_dict[(act_node1, act_node2)][-1]
            if pv_type == 0:
                self.apply_rule(*self.pivot(act_node1, act_node2))
            else:
                self.apply_rule(*self.pivot_gadget(act_node1, act_node2))
            action_id = 2
            node = [act_node1, act_node2]
            
        elif act_type == "GF":
            self.apply_rule(*self.merge_phase_gadgets(act_node1))
            action_id = 6
            node = act_node1
            
        elif act_type == "STOP":
            action_id = 0
            node = [-1]
            
        else:
            action_id = 5  
            reward = 0.0
            node = [-1]
            
       
        # NOTE(cgp): 自己コピーでほかの参照を切ってる、それやるならapply_ruleの前でやれという気持ちもある
        # NOTE(cgp): これは、BaseGraph.copyを呼んでるけど、このメゾットは行儀が悪くてコピーだけしてるわけじゃないことに注意
        self.graph = self.graph.copy() #Relabel nodes due to PYZX not keeping track of node id properly.
        graph = self.graph.copy()
        graph.normalize()

        try:
            circuit = zx.extract_circuit(graph, up_to_perm=True)
            circuit = circuit.to_basic_gates()
            circ = zx.basic_optimization(circuit).to_basic_gates()
            circuit_data = self.get_data(circ)
            new_gates = circuit_data[self.gate_type]
        except Exception as e:
            # NOTE(cgp): ここがnanバグの温床でした。回路が復元できないような操作はSTOPになるんだけど、
            # それにたいしてペナルティを設定したかったのかわからんけど、infだとアドバンテージの計算でnanになる
            # そして、勾配計算もnanになる.
            # なんでdev=cpuで発生するのかはわからんけど... 別にdev=cudaでもここを通れば発生すると思うんだけど
            # new_gates = np.inf
            new_gates = self.current_gates
            print(self.action_pattern, [act_type, new_gates-self.current_gates, vs])
            act_type = "STOP"
            print("error", e)
            # import pickle; pickle.dump(self.init_graph, open("graph.pkl", "wb"))
            # import sys; sys.exit()
        
        self.action_pattern.append([act_type, new_gates-self.current_gates, vs])
        reward = 0
        # NOTE(cgp): エピソード中で最小のゲート数のものを出力とする
        if new_gates < self.min_gates:
            self.min_gates = new_gates
            self.final_circuit = circ            
            
        if new_gates <= self.min_gates:
            self.opt_episode_len = self.episode_len
            self.best_action_stats = copy.deepcopy(self.episode_stats)

        # NOTE(cgp): 報酬の設定 削減率(self.max_compression = 55)
        reward += (self.current_gates - new_gates) / self.max_compression
        self.episode_reward += reward
        # print(self.current_gates, new_gates, reward, self.episode_reward)

        self.pivot_info_dict = self.match_pivot_parallel() | self.match_pivot_boundary() | self.match_pivot_gadget()
        self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
        self.gadget_fusion_ids = list(self.gadget_info_dict)
        # Obtain Length of Remaining Actions:
        remaining_pivot = len(self.pivot_info_dict.keys())
        remaining_lcomp = len(self.match_lcomp())
        remaining_ids = len(self.match_ids())
        remaining_gadget_fusions = len(self.gadget_fusion_ids)
        remaining_actions = remaining_pivot + remaining_lcomp + remaining_ids + remaining_gadget_fusions
        
        history = ActionHistory()
        history.act = act_type
        if 'act_node1' in locals():
            history.vs.append(act_node1)
        if 'act_node2' in locals():
            history.vs.append(act_node2)
        history.gate_reduction = new_gates - self.current_gates
        history.reward = reward

        # print(remaining_actions, " ", end="")

        # End episode if there are no remaining actions or Maximum Length Reached or Incorrect Action Selected
        if (
            remaining_actions == 0 or act_type == "STOP" #or self.episode_len == self.max_episode_len
        ):
            # NOTE(cgp): 重要なSTOP報酬
            if self.args is not None and 'reward' in self.args and self.args.reward == 'sf':
                # straightforward reward
                reward += (min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type])-self.min_gates)/self.max_compression
            elif self.args is not None and 'reward' in self.args and self.args.reward == 'no-stopreward':
                # no stop reward
                pass
            else:
                # 終了時のゲート数と従来手法のゲート数の差
                reward += (min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type])-new_gates)/self.max_compression

            history.reward = reward

            self.current_gates = new_gates
            
            # RL vs PyZX Simplication -> BO, BO, Initial
            if self.min_gates < min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type]):
                win_vs_pyzx = 1
            elif self.min_gates == min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type]):
                win_vs_pyzx = 0
            else:
                win_vs_pyzx = -1
            
            done = True

            if not self.silent:
                print("Win vs Pyzx: ", win_vs_pyzx, " Episode Gates: ", self.min_gates, "Cflow_gates: ", self.pyzx_gates, "Episode Len", self.episode_len, "Opt Episode Len", self.opt_episode_len)
            return (
                self.graph,
                reward,
                done,
                False,
                {
                    "action": action_id,
                    "remaining_lcomp_size": remaining_lcomp,
                    "remaining_pivot_size": remaining_pivot,
                    "remaining_id_size": remaining_ids,
                    "max_reward_difference": self.max_reward,
                    "action_pattern": self.action_pattern,
                    "opt_episode_len": self.opt_episode_len - self.episode_len,
                    "episode_len": self.episode_len,
                    "nstep": self.episode_len,
                    "pyzx_stats": self.pyzx_data,
                    "rl_stats": self.get_data(self.final_circuit),
                    "no_opt_stats": self.no_opt_stats,
                    "swap_cost": self.swap_cost,
                    "pyzx_swap_cost": self.pyzx_swap_cost,
                    "pyzx_gates": self.pyzx_gates,
                    "rl_gates": self.get_data(self.final_circuit)[self.gate_type],
                    "bo_stats": self.basic_opt_data,
                    "initial_stats": self.initial_stats,
                    "win_vs_pyzx": win_vs_pyzx,
                    "min_gates": self.min_gates,
                    #"graph_obs": [self.policy_obs(), self.value_obs()],
                    "final_circuit": self.final_circuit,
                    "final_graph": self.graph,
                    "action_stats": [self.best_action_stats["pivb"], 
                                     self.best_action_stats["pivg"],
                                     self.best_action_stats["piv"],
                                     self.best_action_stats["lc"],
                                     self.best_action_stats["id"],
                                     self.best_action_stats["gf"]],
                    "depth": self.final_circuit.depth(),
                    "initial_depth": self.initial_depth,
                    'history': history,

                    "piv_nodes": self.pivot_info_dict,
                    "lcomp_nodes": self.match_lcomp(),
                    "iden_nodes": self.match_ids(),
                    "gf_nodes": self.gadget_info_dict,

                },
            )



        self.current_gates = new_gates

        return (
            self.graph,
            reward,
            done,
            False,
            {
                "action": action_id,
                "nodes": node,
                #"graph_obs": [self.policy_obs(), self.value_obs()],
                # 前計算
                "piv_nodes": self.pivot_info_dict,
                "lcomp_nodes": self.match_lcomp(),
                "iden_nodes": self.match_ids(),
                "gf_nodes": self.gadget_info_dict,
                "circuit_data": circuit_data,
                'history': history,
                "nstep": self.episode_len,
            },
        )


    def to_graph_like(self):
        """Transforms a ZX-diagram into graph-like"""
        # turn all red spiders into green spiders
        zx.simplify.to_gh(self.graph)
        zx.simplify.spider_simp(self.graph, quiet=True)

    def apply_rule(self, edge_table, rem_vert, rem_edge, check_isolated_vertices):
        self.graph.add_edge_table(edge_table)
        self.graph.remove_edges(rem_edge)
        self.graph.remove_vertices(rem_vert)
        if check_isolated_vertices:
            self.graph.remove_isolated_vertices()

    MatchLcompType = Tuple[VT,Tuple[VT,...]]
    def match_lcomp(self,
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
        if vertexf is not None: candidates = set([v for v in self.graph.vertices() if vertexf(v)])
        else: candidates = self.graph.vertex_set()
        
        phases = self.graph.phases()
        types = self.graph.types()
        
        i = 0
        m: List[MatchLcompType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            v = candidates.pop()
            
            if types[v] != VertexType.Z: continue
            if phases[v] not in (Fraction(1,2), Fraction(3,2)): continue
            if self.graph.is_ground(v): continue

            if check_edge_types and not (
                all(self.graph.edge_type(e) == EdgeType.HADAMARD for e in self.graph.incident_edges(v))
                ): continue

            vn = list(self.graph.neighbors(v))
            if any(types[n] != VertexType.Z for n in vn): continue
            
            #m.append((v,tuple(vn)))
            if len(self.graph.neighbors(v)) ==1:  #Phase gadget of pi/2 can not be selected
                continue
            flag = False
            for neigh_pg in self.graph.neighbors(v): #If root node of phase gadget is a neighbor of candidate node, node can not be selected.
                for neigh_pg2 in self.graph.neighbors(neigh_pg):
                    if len(self.graph.neighbors(neigh_pg2))==1:
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
            candidates = set([e for e in self.graph.edges() if matchf(e)])
        else:
            candidates = self.graph.edge_set()

        types = self.graph.types()
        phases = self.graph.phases()
        matches_dict = {}
        i = 0
        m: List[MatchPivotType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            e = candidates.pop()
            if check_edge_types and self.graph.edge_type(e) != EdgeType.HADAMARD:
                continue

            v0, v1 = self.graph.edge_st(e)
            if not (types[v0] == VertexType.Z and types[v1] == VertexType.Z):
                continue
            if any(phases[v] not in (0, 1) for v in (v0, v1)):
                continue
            if self.graph.is_ground(v0) or self.graph.is_ground(v1):
                continue

            invalid_edge = False
            v0n = list(self.graph.neighbors(v0))
            v0b = []
            for n in v0n:
                if types[n] == VertexType.Z and self.graph.edge_type(self.graph.edge(v0, n)) == EdgeType.HADAMARD:
                    pass
                elif types[n] == VertexType.BOUNDARY:
                    v0b.append(n)
                else:
                    invalid_edge = True
                    break
            if invalid_edge:
                continue

            v1n = list(self.graph.neighbors(v1))
            v1b = []
            for n in v1n:
                if types[n] == VertexType.Z and self.graph.edge_type(self.graph.edge(v1, n)) == EdgeType.HADAMARD:
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

    def match_pivot_gadget(
        self, matchf: Optional[Callable[[ET], bool]] = None, num: int = -1, allow_interacting_matches: bool = False
    ) -> List[MatchPivotType[VT]]:
        """Like :func:`match_pivot_parallel`, but except for pairings of
        Pauli vertices, it looks for a pair of an interior Pauli vertex and an
        interior non-Clifford vertex in order to gadgetize the non-Clifford vertex."""
        if matchf is not None:
            candidates = set([e for e in self.graph.edges() if matchf(e)])
        else:
            candidates = self.graph.edge_set()

        types = self.graph.types()
        phases = self.graph.phases()
        matches_dict = {}
        i = 0
        m: List[MatchPivotType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            e = candidates.pop()
            v0, v1 = self.graph.edge_st(e)
            if not all(types[v] == VertexType.Z for v in (v0, v1)):
                continue

            if phases[v0] not in (0, 1):
                if phases[v1] in (0, 1):
                    v0, v1 = v1, v0
                else:
                    continue
            elif phases[v1] in (0, 1):
                continue  # Now v0 has a Pauli phase and v1 has a non-Pauli phase

            if self.graph.is_ground(v0):
                continue

            v0n = list(self.graph.neighbors(v0))
            v1n = list(self.graph.neighbors(v1))
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
                    ne = list(self.graph.incident_edges(n))
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

    def match_pivot_boundary(
        self, matchf: Optional[Callable[[VT], bool]] = None, num: int = -1, allow_interacting_matches: bool = False
    ) -> List[MatchPivotType[VT]]:
        """Like :func:`match_pivot_parallel`, but except for pairings of
        Pauli vertices, it looks for a pair of an interior Pauli vertex and a
        boundary non-Pauli vertex in order to gadgetize the non-Pauli vertex."""
        if matchf is not None:
            candidates = set([v for v in self.graph.vertices() if matchf(v)])
        else:
            candidates = self.graph.vertex_set()

        phases = self.graph.phases()
        types = self.graph.types()
        matches_dict = {}
        i = 0
        consumed_vertices: Set[VT] = set()
        m: List[MatchPivotType[VT]] = []
        while (num == -1 or i < num) and len(candidates) > 0:
            v = candidates.pop()
            if types[v] != VertexType.Z or phases[v] not in (0, 1) or self.graph.is_ground(v):
                continue

            good_vert = True
            w = None
            bound = None
            for n in self.graph.neighbors(v):
                if (
                    types[n] != VertexType.Z
                    or len(self.graph.neighbors(n)) == 1
                    or n in consumed_vertices
                    or self.graph.is_ground(n)
                ):
                    good_vert = False
                    break

                boundaries = []
                wrong_match = False
                for b in self.graph.neighbors(n):
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

    def lcomp(self, node):
        phase = self.graph.phase(node)
        neighbors = list(self.graph.neighbors(node))
        edge_table = dict()
        vertice = []
        vertice.append(node)
        n = len(neighbors)
        if phase.numerator == 1:
            self.graph.scalar.add_phase(Fraction(1, 4))
        else:
            self.graph.scalar.add_phase(Fraction(7, 4))
        self.graph.scalar.add_power((n - 2) * (n - 1) // 2)
        for i in range(n):
            self.graph.add_to_phase(neighbors[i], -phase)
            for j in range(i + 1, n):
                edge_neigh = self.graph.edge(neighbors[i], neighbors[j])  # edge type between neighbours
                he = edge_table.get(edge_neigh, [0, 0])[1]
                edge_table[edge_neigh] = [0, he + 1]

        return (edge_table, vertice, [], True)

    def pivot(self, v0, v1) -> RewriteOutputType[ET, VT]:
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
        m[2], m[3], _ = self.pivot_info_dict[(v0, v1)]
        phases = self.graph.phases()
        n = [set(self.graph.neighbors(m[0])), set(self.graph.neighbors(m[1]))]
        for i in range(2):
            n[i].remove(m[1 - i])  # type: ignore # Really complex typing situation
            if len(m[i + 2]) == 1:
                n[i].remove(m[i + 2][0])  # type: ignore

        n.append(n[0] & n[1])  #  n[2] <- non-boundary neighbors of m[0] and m[1]
        n[0] = n[0] - n[2]  #  n[0] <- non-boundary neighbors of m[0] only
        n[1] = n[1] - n[2]  #  n[1] <- non-boundary neighbors of m[1] only

        es = (
            [self.graph.edge(s, t) for s in n[0] for t in n[1]]
            + [self.graph.edge(s, t) for s in n[1] for t in n[2]]
            + [self.graph.edge(s, t) for s in n[0] for t in n[2]]
        )
        k0, k1, k2 = len(n[0]), len(n[1]), len(n[2])
        self.graph.scalar.add_power(k0 * k2 + k1 * k2 + k0 * k1)

        for v in n[2]:
            if not self.graph.is_ground(v):
                self.graph.add_to_phase(v, 1)

        if phases[m[0]] and phases[m[1]]:
            self.graph.scalar.add_phase(Fraction(1))
        if not m[2] and not m[3]:
            self.graph.scalar.add_power(-(k0 + k1 + 2 * k2 - 1))
        elif not m[2]:
            self.graph.scalar.add_power(-(k1 + k2))
        else:
            self.graph.scalar.add_power(-(k0 + k2))

        for i in range(2):  # if m[i] has a phase, it will get copied on to the neighbors of m[1-i]:
            a = phases[m[i]]  # type: ignore
            if a:
                for v in n[1 - i]:
                    if not self.graph.is_ground(v):
                        self.graph.add_to_phase(v, a)
                for v in n[2]:
                    if not self.graph.is_ground(v):
                        self.graph.add_to_phase(v, a)

            if not m[i + 2]:
                rem_verts.append(m[1 - i])  # type: ignore # if there is no boundary, the other vertex is destroyed
            else:
                e = self.graph.edge(m[i], m[i + 2][0])  # type: ignore # if there is a boundary, toggle whether it is an h-edge or a normal edge
                new_e = self.graph.edge(m[1 - i], m[i + 2][0])  # type: ignore # and point it at the other vertex
                ne, nhe = etab.get(new_e, [0, 0])
                if self.graph.edge_type(e) == EdgeType.SIMPLE:
                    nhe += 1
                elif self.graph.edge_type(e) == EdgeType.HADAMARD:
                    ne += 1
                etab[new_e] = [ne, nhe]
                rem_edges.append(e)

        for e in es:
            nhe = etab.get(e, (0, 0))[1]
            etab[e] = [0, nhe + 1]

        return (etab, rem_verts, rem_edges, True)

    def pivot_gadget(self, v0, v1) -> RewriteOutputType[ET, VT]:
        """Performs the gadgetizations required before applying pivots.
        ``m[0]`` : interior pauli vertex
        ``m[1]`` : interior non-pauli vertex to gadgetize
        ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
        ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
        """
        vertices_to_gadgetize = v1
        self.gadgetize(vertices_to_gadgetize)
        return self.pivot(v0, v1)

    def gadgetize(self, vertices: VT) -> None:
        """Helper function which pulls out a list of vertices into gadgets"""
        edge_list = []

        inputs = self.graph.inputs()
        phases = self.graph.phases()
        v = vertices
        if any(n in inputs for n in self.graph.neighbors(v)):
            mod = 0.5
        else:
            mod = -0.5

        vp = self.graph.add_vertex(VertexType.Z, -2, self.graph.row(v) + mod, phases[v])
        v0 = self.graph.add_vertex(VertexType.Z, -1, self.graph.row(v) + mod, 0)
        self.graph.set_phase(v, 0)

        edge_list.append(self.graph.edge(v, v0))
        edge_list.append(self.graph.edge(v0, vp))

        if self.graph.phase_tracking:
            self.graph.unfuse_vertex(vp, v)

        self.graph.add_edges(edge_list, EdgeType.HADAMARD)
        return

    MatchGadgetType = Tuple[VT, int, List[VT], Dict[VT, VT]]

    def match_phase_gadgets(self, vertexf: Optional[Callable[[VT], bool]] = None) -> List[MatchGadgetType[VT]]:
        """Determines which phase gadgets act on the same vertices, so that they can be fused together.

        :param g: An instance of a ZX-graph.
        :rtype: List of 4-tuples ``(leaf, parity_length, other axels with same targets, leaf dictionary)``.
        1.leaf is a vertex that represents a phase gadget
        2.parity_length is the number of vertices that the phase gadget acts on
        3.other_axels is a list of other phase gadgets that act on the same vertices as leaf
        4.leaf_dict is a dictionary that maps each phase gadget to its corresponding phase node
        """
        if vertexf is not None:
            candidates = set([v for v in self.graph.vertices() if vertexf(v)])
        else:
            candidates = self.graph.vertex_set()
        gadget_info_dict = {}
        phases = self.graph.phases()

        parities: Dict[FrozenSet[VT], List[VT]] = dict()
        gadgets: Dict[VT, VT] = dict()
        inputs = self.graph.inputs()
        outputs = self.graph.outputs()
        # First we find all the phase-gadgets, and the list of vertices they act on
        for v in candidates:
            non_clifford = phases[v] != 0 and getattr(phases[v], "denominator", 1) > 2
            if isinstance(phases[v], Poly):
                non_clifford = True
            if non_clifford and len(list(self.graph.neighbors(v))) == 1:
                n = list(self.graph.neighbors(v))[0]
                if phases[n] not in (0, 1):
                    continue  # Not a real phase gadget (happens for scalar diagrams)
                if n in gadgets:
                    continue  # Not a real phase gadget (happens for scalar diagrams)
                if n in inputs or n in outputs:
                    continue  # Not a real phase gadget (happens for non-unitary diagrams)
                gadgets[n] = v
                par = frozenset(set(self.graph.neighbors(n)).difference({v}))
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

    def merge_phase_gadgets(self, vertexs: Tuple[VT]) -> RewriteOutputType[ET, VT]:
        """v0,v1"""
        """Given the output of :func:``match_phase_gadgets``, removes phase gadgets that act on the same set of targets."""
        rem = []
        phases = self.graph.phases()
        par_num = self.gadget_info_dict[vertexs]
        n = vertexs[0]
        gad = list(vertexs[1:])
        gadgets = self.gadgets

        v = gadgets[n]
        if len(gad) == 0:
            if phases[n] != 0:
                self.graph.scalar.add_phase(phases[v])
                if self.graph.phase_tracking:
                    self.graph.phase_negate(v)
                phase = -phases[v]
        else:
            phase = sum((1 if phases[w] == 0 else -1) * phases[gadgets[w]] for w in gad + [n]) % 2
            for w in gad + [n]:
                if phases[w] != 0:
                    self.graph.scalar.add_phase(phases[gadgets[w]])
                    if self.graph.phase_tracking:
                        self.graph.phase_negate(gadgets[w])
            self.graph.scalar.add_power(-((par_num - 1) * len(gad)))
        self.graph.set_phase(v, phase)
        self.graph.set_phase(n, 0)
        othertargets = [gadgets[w] for w in gad]
        rem.extend(gad)
        rem.extend(othertargets)
        for w in othertargets:
            if self.graph.phase_tracking:
                self.graph.fuse_phases(v, w)
            if self.graph.merge_vdata is not None:
                self.graph.merge_vdata(v, w)
        return ({}, rem, [], False)

    def spider_fusion(self, neighs):
        rem_verts = []
        etab = dict()

        if self.graph.row(neighs[0]) == 0:
            v0, v1 = neighs[1], neighs[0]
        else:
            v0, v1 = neighs[0], neighs[1]
        ground = self.graph.is_ground(v0) or self.graph.is_ground(v1)
        if ground:
            self.graph.set_phase(v0, 0)
            self.graph.set_ground(v0)
        else:
            self.graph.add_to_phase(v0, self.graph.phase(v1))
        if self.graph.phase_tracking:
            self.graph.fuse_phases(v0, v1)
        # always delete the second vertex in the match
        rem_verts.append(v1)
        # edges from the second vertex are transferred to the first
        for w in self.graph.neighbors(v1):
            if v0 == w:
                continue
            e = self.graph.edge(v0, w)
            if e not in etab:
                etab[e] = [0, 0]
            etab[e][self.graph.edge_type(self.graph.edge(v1, w)) - 1] += 1
        return (etab, rem_verts, [], True)

    def remove_ids(self, node):
        neigh = self.graph.neighbors(node)
        v0, v1 = neigh
        if self.graph.edge_type(self.graph.edge(node, v0)) != self.graph.edge_type(
            self.graph.edge(node, v1)
        ):  # exactly one of them is a hadamard edge
            et = zx.EdgeType.HADAMARD
        else:
            et = zx.EdgeType.SIMPLE
        # create dict, rem_vertexs
        etab = dict()
        e = self.graph.edge(v0, v1)
        if not e in etab:
            etab[e] = [0, 0]
        if et == zx.EdgeType.SIMPLE:
            etab[e][0] += 1
        else:
            etab[e][1] += 1
        return (etab, [node], [], False)

    def match_ids(self):
        candidates = self.graph.vertex_set()
        types = self.graph.types()
        phases = self.graph.phases()
        m = []
        while len(candidates) > 0:
            v = candidates.pop()
            if phases[v] != 0 or not zx.utils.vertex_is_zx(types[v]) or self.graph.is_ground(v):
                continue
            neigh = self.graph.neighbors(v)
            if len(neigh) != 2:
                continue
            v0, v1 = neigh
            if (
                self.graph.is_ground(v0)
                and types[v1] == zx.VertexType.BOUNDARY
                or self.graph.is_ground(v1)
                and types[v0] == zx.VertexType.BOUNDARY
            ):
                # Do not put ground spiders on the boundary
                continue
            m.append(v)
        return m

    def get_data(self, circuit):
        clifford = 0
        hadamards = 0
        twoqubits = 0
        cnots = 0
        cz = 0
        cx = 0
        tcount = 0
        total = 0
        for g in circuit.gates:
            total += 1
            tcount += g.tcount()
            if isinstance(g, (zx.gates.ZPhase, zx.gates.XPhase)):
                if g.phase.denominator <= 2:
                    clifford += 1

            elif isinstance(g, (zx.gates.HAD)):
                hadamards += 1
                clifford += 1
            elif isinstance(g, (zx.gates.CZ, zx.gates.CNOT)):
                twoqubits += 1
                if isinstance(g, zx.gates.CNOT):
                    cnots += 1
                elif isinstance(g, zx.gates.CZ):
                    cz += 1

        d = dict()
        d["qubits"] = circuit.qubits
        d["gates"] = total
        d["tcount"] = tcount
        d["clifford"] = clifford
        d["CNOT"] = cnots
        d["CZ"] = cz
        d["had"] = hadamards
        d["twoqubits"] = twoqubits

        return d

    def obtain_gates_pyzx(self, g):
        graph = g.copy()
        zx.to_graph_like(graph)
        zx.flow_2Q_simp(graph)
        circuit = zx.extract_simple(graph).to_basic_gates()

        circuit = zx.basic_optimization(circuit).to_basic_gates()
        self.pyzx_swap_cost = 0
        return self.get_data(circuit)
    
    # 外部から操作するための関数、Envは内部でinfoを使っているので
    def set_info(self, info):
        self.pivot_info_dict = self.match_pivot_parallel() | self.match_pivot_boundary() | self.match_pivot_gadget()
        self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
        self.gadget_fusion_ids = list(self.gadget_info_dict)
    
    def get_info(self):
        self.pivot_info_dict = self.match_pivot_parallel() | self.match_pivot_boundary() | self.match_pivot_gadget()
        self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
        self.gadget_fusion_ids = list(self.gadget_info_dict)
        return {
            "piv_nodes": self.pivot_info_dict,
            "lcomp_nodes": self.match_lcomp(),
            "iden_nodes": self.match_ids(),
            "gf_nodes": self.gadget_info_dict,
            "nstep": self.episode_len,
        }


class ZXEnv(ZXEnvBase):
    def __init__(self, qubits, depth, gate_type, silent:bool = False, args: argparse.Namespace = None):
        self.qubits, self.depth = qubits, depth
        self.gate_type = gate_type
        super().__init__(silent=silent, args=args)
    
    def reset(self):
        # parameters
        self.episode_len = 0
        self.episode_reward = 0
        self.action_pattern = []
        self.max_reward = 0
        self.opt_episode_len = 0
        self.min_gates = self.depth
        self.swap_cost = 0
        self.episode_stats = {"pivb": 0 , "pivg":0, "piv":0, "lc": 0, "id":0, "gf":0}
        self.best_action_stats = {"pivb": 0 , "pivg":0, "piv":0 , "lc": 0, "id":0, "gf":0}
        valid_circuit = False
        
        # circuit generation
        while not valid_circuit:
            def get_nx_graph(g):
                nodelist = []
                for v in g.vertices():
                    nodelist.append([v, {'phase':g.phases()[v]}])
                edgelist = []
                for v1, v2 in g.edges():
                    edgelist.append([v1, v2, {'edge_type': g.graph[v1][v2]}])
                G = nx.Graph()
                G.add_nodes_from(nodelist)
                G.add_edges_from(edgelist)
                return G

            # print()
            # print("generating random graph")
            # print_random_states(show_hash=True)

            g = zx.generate.cliffordT(
               self.qubits, self.depth, p_t=0.17, p_s=0.24, p_hsh=0.25, 
            )
            self.init_graph = g.copy()

            # print(nx.weisfeiler_lehman_graph_hash(get_nx_graph(g)))
            c = zx.Circuit.from_graph(g)
            self.no_opt_stats = self.get_data(c.to_basic_gates())
            self.initial_depth = c.to_basic_gates().depth()
            self.rand_circuit = zx.optimize.basic_optimization(c.split_phase_gates())
            self.initial_stats = self.get_data(self.rand_circuit)
            self.graph = self.rand_circuit.to_graph()
            
            # signal.signal(signal.SIGALRM, handler)
            # signal.alarm(10)
            try:
                zx.simplify.teleport_reduce(self.graph)
            except: 
                print('Teleport reduce error')
                continue
            
            # signal.alarm(0)
            
            basic_circ = zx.optimize.basic_optimization(zx.Circuit.from_graph(self.graph.copy()).split_phase_gates())
            self.basic_opt_data = self.get_data(basic_circ.to_basic_gates())
            self.to_graph_like()
            self.graph = self.graph.copy()  # This relabels the nodes such that there are no empty spaces
            
            
            self.pivot_info_dict = self.match_pivot_parallel() | self.match_pivot_boundary() | self.match_pivot_gadget()
            self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
            self.gadget_fusion_ids = list(self.gadget_info_dict)
            match_lcomp = self.match_lcomp()
            match_ids = self.match_ids()
            actions_available = len(self.match_lcomp()) + len(self.pivot_info_dict.keys()) + len(self.match_ids())
            if actions_available == 0:
                print("Generating new circuit")
            else:
                valid_circuit = True
            
            full_reduce_start = time.time()
            self.pyzx_data = self.obtain_gates_pyzx(g.copy())
            full_reduce_end = time.time()

            self.pyzx_gates = self.pyzx_data[self.gate_type]
            circuit= zx.extract_circuit(self.graph.copy(), up_to_perm=True)
            circuit = circuit.to_basic_gates()
            circuit = zx.basic_optimization(circuit).to_basic_gates()
            circuit_data = self.get_data(circuit)
            self.current_gates = circuit_data[self.gate_type]
            self.initial_stats = circuit_data
            self.final_circuit = circuit
            self.min_gates = circuit_data[self.gate_type]

        return self.graph, {
            "graph_obs": self.graph,
            "full_reduce_time": full_reduce_end-full_reduce_start,
            "piv_nodes": self.pivot_info_dict,
            "lcomp_nodes": match_lcomp,
            "iden_nodes": self.match_ids(),
            "gf_nodes": self.gadget_info_dict,
            "circuit_data": circuit_data,
            "nstep": self.episode_len,
        }

class ZXEnvForTest(ZXEnvBase):
    def __init__(self, g, gate_type, silent:bool=False, args: argparse.Namespace=None):
        self.g = g
        self.gate_type = gate_type
        super().__init__(silent=silent, args=args)
    
    def reset(self):
        # parameters
        self.episode_len = 0
        self.episode_reward = 0
        self.action_pattern = []
        self.max_reward = 0
        self.opt_episode_len = 0
        self.min_gates = self.g.depth()
        self.swap_cost = 0
        self.episode_stats = {"pivb": 0 , "pivg":0, "piv":0, "lc": 0, "id":0, "gf":0}
        self.best_action_stats = {"pivb": 0 , "pivg":0, "piv":0 , "lc": 0, "id":0, "gf":0}
        valid_circuit = False
        
        # circuit generation
        while not valid_circuit:
            c = zx.Circuit.from_graph(self.g)
            self.no_opt_stats = self.get_data(c.to_basic_gates())
            self.initial_depth = c.to_basic_gates().depth()
            self.rand_circuit = zx.optimize.basic_optimization(c.split_phase_gates())
            self.initial_stats = self.get_data(self.rand_circuit)
            self.graph = self.rand_circuit.to_graph()
            
            # signal.signal(signal.SIGALRM, handler)
            # signal.alarm(10)
            try:
                zx.simplify.teleport_reduce(self.graph)
            except: 
                print('Teleport reduce error')
                continue
            
            # signal.alarm(0)
            
            basic_circ = zx.optimize.basic_optimization(zx.Circuit.from_graph(self.graph.copy()).split_phase_gates())
            self.basic_opt_data = self.get_data(basic_circ.to_basic_gates())
            self.to_graph_like()
            self.graph = self.graph.copy()  # This relabels the nodes such that there are no empty spaces
            
            
            self.pivot_info_dict = self.match_pivot_parallel() | self.match_pivot_boundary() | self.match_pivot_gadget()
            self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
            match_lcomp = self.match_lcomp()
            self.gadget_fusion_ids = list(self.gadget_info_dict)
            actions_available = len(match_lcomp) + len(self.pivot_info_dict.keys()) + len(self.match_ids())
            
            if actions_available == 0:
                print("Generating new circuit")
                raise Exception("No actions available")
            else:
                valid_circuit = True

            full_reduce_start = time.time()
            self.pyzx_data = self.obtain_gates_pyzx(self.g.copy())
            full_reduce_end = time.time()

            self.pyzx_gates = self.pyzx_data[self.gate_type]
            circuit= zx.extract_circuit(self.graph.copy(), up_to_perm=True)
            circuit = circuit.to_basic_gates()
            circuit = zx.basic_optimization(circuit).to_basic_gates()
            circuit_data = self.get_data(circuit)
            self.current_gates = circuit_data[self.gate_type]
            self.initial_stats = circuit_data
            self.final_circuit = circuit
            self.min_gates = circuit_data[self.gate_type]

        return self.graph, {
            "graph_obs": self.graph,
            "full_reduce_time": full_reduce_end-full_reduce_start,
            "piv_nodes": self.pivot_info_dict,
            "lcomp_nodes": match_lcomp,
            "iden_nodes": self.match_ids(),
            "gf_nodes": self.gadget_info_dict,
            "circuit_data": circuit_data,
            "nstep": self.episode_len,
        }


# 少し早いczx実装を使うmixin
class CZXStepMixin():
    def manual_step(self, act_type, vs):
        if len(vs) == 1:
            act_node1 = vs[0]
        elif len(vs) == 2:
            act_node1 = vs[0]
            act_node2 = vs[1]


        # Update Stats
        self.render_flag = 1
        self.episode_len += 1
        reward = 0
        done = False
        

        if act_type == "LC":
            
            self.apply_rule(*self.lcomp(act_node1))
            action_id = 1
            node = [act_node1]
            
        elif act_type == "ID":
            
            neighbours = list(self.graph.neighbors(act_node1))
            types = self.graph.types()
            self.apply_rule(*self.remove_ids(act_node1))
            if types[neighbours[0]] != zx.VertexType.BOUNDARY and types[neighbours[1]] != zx.VertexType.BOUNDARY:
                self.apply_rule(*self.spider_fusion(neighbours))
            action_id = 3
            node = [act_node1]
            
        elif act_type == "PV" or act_type == "PVG" or act_type == "PVB":
            
            pv_type = self.pivot_info_dict[(act_node1, act_node2)][-1]
            if pv_type == 0:
                self.apply_rule(*self.pivot(act_node1, act_node2))
            else:
                self.apply_rule(*self.pivot_gadget(act_node1, act_node2))
            action_id = 2
            node = [act_node1, act_node2]
            
        elif act_type == "GF":
            self.apply_rule(*self.merge_phase_gadgets(act_node1))
            action_id = 6
            node = act_node1
            
        elif act_type == "STOP":
            action_id = 0
            node = [-1]
            
        else:
            action_id = 5  
            reward = 0.0
            node = [-1]
            
       
        # NOTE(cgp): 自己コピーでほかの参照を切ってる、それやるならapply_ruleの前でやれという気持ちもある
        # NOTE(cgp): これは、BaseGraph.copyを呼んでるけど、このメゾットは行儀が悪くてコピーだけしてるわけじゃないことに注意
        self.graph = self.graph.copy() #Relabel nodes due to PYZX not keeping track of node id properly.
        graph = self.graph.copy()
        graph.normalize()
        
        try:
            import czx_nano.czx_ext as czx
            circuit_data = czx.extract_and_basic_optimization(graph)
            new_gates = circuit_data[self.gate_type]
        except Exception as e:
            # NOTE(cgp): ここがnanバグの温床でした。回路が復元できないような操作はSTOPになるんだけど、
            # それにたいしてペナルティを設定したかったのかわからんけど、infだとアドバンテージの計算でnanになる
            # そして、勾配計算もnanになる.
            # なんでdev=cpuで発生するのかはわからんけど... 別にdev=cudaでもここを通れば発生すると思うんだけど
            # new_gates = np.inf
            new_gates = self.current_gates
            print(self.action_pattern, [act_type, new_gates-self.current_gates, vs])
            act_type = "STOP"
            print("error", e)
            # import pickle; pickle.dump(self.init_graph, open("graph.pkl", "wb"))
            # import sys; sys.exit()
        
        self.action_pattern.append([act_type, new_gates-self.current_gates, vs])
        reward = 0
        # NOTE(cgp): エピソード中で最小のゲート数のものを出力とする
        if new_gates < self.min_gates:
            self.min_gates = new_gates
            self.final_circuit_data = circuit_data            
            
        if new_gates <= self.min_gates:
            self.opt_episode_len = self.episode_len
            self.best_action_stats = copy.deepcopy(self.episode_stats)

        # NOTE(cgp): 報酬の設定 削減率(self.max_compression = 55)
        reward += (self.current_gates - new_gates) / self.max_compression
        self.episode_reward += reward
        # print(self.current_gates, new_gates, reward, self.episode_reward)

        self.pivot_info_dict = self.match_pivot_parallel() | self.match_pivot_boundary() | self.match_pivot_gadget()
        self.gadget_info_dict, self.gadgets = self.match_phase_gadgets()
        self.gadget_fusion_ids = list(self.gadget_info_dict)
        # Obtain Length of Remaining Actions:
        remaining_pivot = len(self.pivot_info_dict.keys())
        remaining_lcomp = len(self.match_lcomp())
        remaining_ids = len(self.match_ids())
        remaining_gadget_fusions = len(self.gadget_fusion_ids)
        remaining_actions = remaining_pivot + remaining_lcomp + remaining_ids + remaining_gadget_fusions
        
        history = ActionHistory()
        history.act = act_type
        if 'act_node1' in locals():
            history.vs.append(act_node1)
        if 'act_node2' in locals():
            history.vs.append(act_node2)
        history.gate_reduction = new_gates - self.current_gates
        history.reward = reward

        # print(remaining_actions, " ", end="")

        # End episode if there are no remaining actions or Maximum Length Reached or Incorrect Action Selected
        if (
            remaining_actions == 0 or act_type == "STOP" #or self.episode_len == self.max_episode_len
        ):
            # NOTE(cgp): 重要なSTOP報酬
            if self.args is not None and 'reward' in self.args and self.args.reward == 'sf':
                # straightforward reward
                reward += (min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type])-self.min_gates)/self.max_compression
            elif self.args is not None and 'reward' in self.args and self.args.reward == 'no-stopreward':
                # no stop reward
                pass
            elif self.args is not None and 'agent' in self.args and (self.args.agnet == 'ppo-ethernal' or self.args.agent == 'ppg-ethernal'):
                # no stop reward
                pass
            else:
                # 終了時のゲート数と従来手法のゲート数の差
                reward += (min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type])-new_gates)/self.max_compression

            history.reward = reward

            self.current_gates = new_gates
            
            # RL vs PyZX Simplication -> BO, BO, Initial
            if self.min_gates < min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type]):
                win_vs_pyzx = 1
            elif self.min_gates == min(self.pyzx_gates, self.basic_opt_data[self.gate_type], self.initial_stats[self.gate_type]):
                win_vs_pyzx = 0
            else:
                win_vs_pyzx = -1
            
            done = True

            if not self.silent:
                print("Win vs Pyzx: ", win_vs_pyzx, " Episode Gates: ", self.min_gates, "Cflow_gates: ", self.pyzx_gates, "Episode Len", self.episode_len, "Opt Episode Len", self.opt_episode_len)
            return (
                self.graph,
                reward,
                done,
                False,
                {
                    "action": action_id,
                    "remaining_lcomp_size": remaining_lcomp,
                    "remaining_pivot_size": remaining_pivot,
                    "remaining_id_size": remaining_ids,
                    "max_reward_difference": self.max_reward,
                    "action_pattern": self.action_pattern,
                    "opt_episode_len": self.opt_episode_len - self.episode_len,
                    "episode_len": self.episode_len,
                    "nstep": self.episode_len,
                    "pyzx_stats": self.pyzx_data,
                    "rl_stats": self.final_circuit_data,
                    "no_opt_stats": self.no_opt_stats,
                    "swap_cost": self.swap_cost,
                    "pyzx_swap_cost": self.pyzx_swap_cost,
                    "pyzx_gates": self.pyzx_gates,
                    "rl_gates": self.final_circuit_data[self.gate_type],
                    "bo_stats": self.basic_opt_data,
                    "initial_stats": self.initial_stats,
                    "win_vs_pyzx": win_vs_pyzx,
                    "min_gates": self.min_gates,
                    #"graph_obs": [self.policy_obs(), self.value_obs()],
                    # "final_circuit": self.final_circuit,
                    "action_stats": [self.best_action_stats["pivb"], 
                                     self.best_action_stats["pivg"],
                                     self.best_action_stats["piv"],
                                     self.best_action_stats["lc"],
                                     self.best_action_stats["id"],
                                     self.best_action_stats["gf"]],
                    "depth": self.final_circuit_data['depth'],
                    "initial_depth": self.initial_depth,
                    'history': history

                },
            )



        self.current_gates = new_gates

        return (
            self.graph,
            reward,
            done,
            False,
            {
                "action": action_id,
                "nodes": node,
                #"graph_obs": [self.policy_obs(), self.value_obs()],
                # 前計算
                "piv_nodes": self.pivot_info_dict,
                "lcomp_nodes": self.match_lcomp(),
                "iden_nodes": self.match_ids(),
                "gf_nodes": self.gadget_info_dict,
                "circuit_data": circuit_data,
                'history': history,
                "nstep": self.episode_len,
            },
        )

class CZXEnv(CZXStepMixin, ZXEnv):
    pass

class CZXEnvForTest(CZXStepMixin, ZXEnvForTest):
    pass