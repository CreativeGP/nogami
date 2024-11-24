import sys; sys.path.insert(0, '../..') # So that we import the local copy of pyzx if you have installed from Github
import random
from copy import deepcopy
import json
import time
from abc import ABC
from abc import abstractmethod
import pickle
from engineering_notation import EngNumber

import networkx as nx
import pyzx as zx
import numpy as np
import matplotlib.pyplot as plt

import torch
import rl_agent
import zx_env

args = None

def riu_preprocess(g):
    circ = zx.Circuit.from_graph(g)
    circ = circ.split_phase_gates()
    circ = zx.optimize.basic_optimization(circ)
    g = circ.to_graph()

    zx.simplify.teleport_reduce(g)
    
    circ = zx.Circuit.from_graph(g)
    circ = circ.split_phase_gates()
    circ = zx.optimize.basic_optimization(circ)
    g = circ.to_graph()

    zx.simplify.to_gh(g)
    zx.simplify.spider_simp(g)
    return g

def format_float(value, precision, unit):
    return str(EngNumber(value=value, precision=precision))+unit

def dice(p):
    return random.random() < p

def time_lambda(self, str, l):
    import time
    start = time.time()
    l()    
    print(str+ ": ", round(time.time() - start,2) ,"s")



class Timer():
    def __init__(self, silent=False):
        import time
        self.last = time.time()
        self.start = time.time()
        self.silent = silent
    
    def cut(self, point="", silent=False):
        import time
        if not (self.silent or silent): 
            print("[Timer]", point, format_float(time.time() - self.last,2,'s'), format_float(time.time() - self.start,2,'s'))
        self.last = time.time()

class Histogram():
    def __init__(self, name=""):
        self.data = {}
        self.name = name
    
    def add(self, dat):
        self.data[dat] = self.data.get(dat, 0) + 1
    
    def print(self, percentage=False):
        print("Histogram of " + self.name)
        key_list = list(sorted(self.data.keys()))
        mi = min(key_list)
        ma = max(key_list)
        class_value = (ma-mi)//10 + 1
        # if class_value == 0:
        #     class_value = 1
        tmp = np.zeros(10+1)
        for key in sorted(self.data.keys()):
            tmp[(key-mi)//class_value] += self.data[key]
        for i in range(len(tmp)):
            if percentage:
                print(f"{i*class_value+mi} - {(i+1)*class_value+mi} : {tmp[i]/sum(tmp)*100:.2f}%")
            else:
                print(f"{i*class_value+mi} - {(i+1)*class_value+mi} : {tmp[i]:.3f}")

class Scatter():
    def __init__(self, name=""):
        self.name = name
        self.data = []
        self.load()
    
    def add(self, x, y):
        self.data.append((x,y))

    def print(self):
        plt.scatter(*zip(*self.data), alpha=0.05)
        plt.title(self.name)
    
    def save(self):
        pickle.dump(self.data, open(self.name + ".pkl", "wb"))
        
    def load(self):
        # if file exists
        try:
            self.data = pickle.load(open(self.name + ".pkl", "rb"))
        except Exception as e:
            print(e)

scatter_id = Scatter("id")
scatter_lcomp = Scatter("lcomp")
scatter_pivot = Scatter("pivot")



class TraversalOptimizer(ABC):
    def __init__(self, circ):
        self.g = deepcopy(circ)
        self.g_circ = deepcopy(circ)

        self.g = riu_preprocess(self.g)
        # zx.simplify.spider_simp(self.g)
        # zx.simplify.to_gh(self.g)

        print()
        print()
        print("  ================== "+self.__class__.__name__ + "==================")
        print()
        self.actions = {}

    @abstractmethod
    def score(self, g):
        pass


    @abstractmethod
    def should_explore(self, depth, new_score, current_score):
        pass

    def get_nx_graph(self, g):
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

    def run(self):
        min_depth = 10000
        self.min_score = 10000
        self.min_g = None
        min_history = ""
        self.score_histories = []


        # remove boundary vertices
        ids = zx.rules.match_ids_parallel(self.g) # (identity_vertex, neighbor1, neighbor2, edge_type)
        for id_match in ids:
            if not id_match[0] in [5, 6, 7, 8, 9, 50, 51, 52, 53, 54]:
                continue
            etab, rem_verts, rem_edges, check_isolated_vertices = zx.rules.remove_ids(self.g, [id_match])
            try:
                before = self.g.num_vertices()
                self.g.add_edge_table(etab)
                self.g.remove_edges(rem_edges)
                self.g.remove_vertices(rem_verts)
                if check_isolated_vertices:
                    self.g.remove_isolated_vertices()
            except Exception as e:
                print(e)

        database = []
        queue = [(self.g,0,"",[self.score(self.g)])]
        count = 0
        ids_hist = Histogram("ids")
        lcomps_hist = Histogram("lcomps")
        pivots_hist = Histogram("pivots")
        depthwise_hist = {}
        gates_hist = Histogram("gates")
        visited = dict()
        while len(queue) != 0:
            g1, depth, history, score_history = queue.pop(-1)
            current_score = score_history[-1]
            count += 1

            if not self.check_identity(g1):
                print("Identity check: ", self.check_identity(g1), history)

            if not depth in depthwise_hist:
                depthwise_hist[depth] = Histogram("depth " + str(depth))
                depthwise_hist[depth].add(current_score)
            else:
                depthwise_hist[depth].add(current_score)

            graph_hash = nx.weisfeiler_lehman_graph_hash(self.get_nx_graph(g1))
            if graph_hash in visited:
                self.score_histories.append(score_history)
                database.append(g1)
                continue
            else:
                # zx.draw_matplotlib(g1,labels=True,h_edge_draw='box').savefig("current.png")
                visited[graph_hash] = g1
                
            # print("Queue length:", len(queue), history, current_score, )

            branch = 0
            for action in self.actions:
                for match in self.actions[action]["match"](g1):
                    branch += 1
                    g2 = deepcopy(g1)
                    etab, rem_verts, rem_edges, check_isolated_vertices = self.actions[action]["apply"](g2, [match])
                    try:
                        before = g2.num_vertices()
                        g2.add_edge_table(etab)
                        g2.remove_edges(rem_edges)
                        g2.remove_vertices(rem_verts)
                        if check_isolated_vertices:
                            g2.remove_isolated_vertices()

                        new_score = self.score(g2)
                        if self.should_explore(depth, new_score, current_score):
                            queue.append((g2,depth+1,history + f" {self.actions[action]['name']}({new_score - current_score})", score_history + [new_score]))
                        after = g2.num_vertices()
                        #print("id ", before, after)
                        if self.actions[action]["name"] == "id":
                            scatter_id.add(len(g1.neighbors(match[0])), new_score-current_score)
                        elif self.actions[action]["name"] == "lcomp":
                            scatter_lcomp.add(len(g1.neighbors(match[0])), new_score-current_score)
                        elif self.actions[action]["name"] == "pivot":
                            scatter_pivot.add(len(g1.neighbors(match[0])), new_score-current_score)

                        gates_hist.add(new_score)
                        if  new_score <= self.min_score:
                            self.min_score = new_score
                            min_history = history + f" {self.actions[action]['name']}({new_score - current_score})"
                            self.min_g = g2
                            # print(self.min_score, min_history)
                    except Exception as e:
                        print(e)

            if branch == 0:
                self.score_histories.append(score_history)
                database.append(g1)


        # databases = []
        # try:
        #     left = pickle.load(open("database.pkl", "rb"))
        #     databases.extend(left)
        # except Exception as e:
        #     print(e)
        # databases.append(database)
        # pickle.dump(databases, open("database.pkl", "wb"))
        
        print("Min score: ", self.min_score)
        print("Min history: ", min_history)
        print("Traversal count: ", count)

        #self.save_image()
    
    def check_identity(self, argg):
        _argg = argg.copy()
        #zx.simplify.to_graph_like(_argg)
        #zx.simplify.flow_2Q_simp(graph)
        #zx.simplify.flow_2Q_simp(graph)
        # circuit = zx.extract.extract_clifford_normal_form(_argg).to_basic_gates()
        circuit = zx.extract.extract_circuit(_argg).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        return zx.compare_tensors(circuit.to_tensor(),self.g.to_tensor())

    def save_image(self):
        #circ = zx.extract.extract_clifford_normal_form(self.min_g).to_basic_gates()
        circ = zx.extract.extract_circuit(self.min_g).to_basic_gates()
        circ = zx.basic_optimization(circ)
        self.draw_2_graphs(self.g_circ, circ, "before-after.png")
    
    def draw_2_graphs(self, g1, g2, filename):
        fig = zx.draw_matplotlib(g1,labels=True,h_edge_draw='box')
        # circ = zx.basic_optimization(g_circ).to_basic_gates()
        fig.savefig("g1.png")
        fig = zx.draw_matplotlib(g2,labels=True,h_edge_draw='box')
        fig.savefig("g2.png")

        # concatenate images
        from PIL import Image
        im1 = Image.open("g1.png")
        im2 = Image.open("g2.png")
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        dst.save(filename)
        
        # remove before.png and after.png
        import os
        os.remove("g1.png")
        os.remove("g2.png")
        

    def plot_traversal_graph(self):
        for i, history in enumerate(self.score_histories):
            x = list(range(1,len(history)+1))
            plt.plot(x, history, alpha=0.05)
        plt.show()



class Optimizer1(TraversalOptimizer):
    def __init__(self, circ):
        super().__init__(circ)
                #  Action: id, pivot, pivot_boudary, pivot_gadget, lcomp, gadget fusion

    def run(self):
        min_depth = 10000
        self.min_score = 10000
        self.min_g = None
        min_history = ""
        self.score_histories = []


        # remove boundary vertices
        ids = zx.rules.match_ids_parallel(self.g) # (identity_vertex, neighbor1, neighbor2, edge_type)
        for id_match in ids:
            if not id_match[0] in [5, 6, 7, 8, 9, 50, 51, 52, 53, 54]:
                continue
            etab, rem_verts, rem_edges, check_isolated_vertices = zx.rules.remove_ids(self.g, [id_match])
            try:
                before = self.g.num_vertices()
                self.g.add_edge_table(etab)
                self.g.remove_edges(rem_edges)
                self.g.remove_vertices(rem_verts)
                if check_isolated_vertices:
                    self.g.remove_isolated_vertices()
            except Exception as e:
                print(e)

        self.actions = {
            "id": {
                "name": "id",
                "match": zx_env.match_ids,
                "apply": zx_env.remove_ids,
            },
            "lcomp": {
                "name": "lcomp",
                "match": riu_zxenv.match_lcomp,
                "apply": riu_zxenv.lcomp,
            },
            "pivot": {
                "name": "pivot",
                "match": riu_zxenv.match_pivot_parallel,
                "apply": riu_zxenv.pivot,
            },
            # "pivot_boundary": {
            #     "name": "pivot_boundary",
            #     "match": zx.rules.match_pivot_boundary,
            #     "apply": zx.rules.pivot_boundary,
            # },
            # "pivot_gadget": {
            #     "name": "pivot_gadget",
            #     "match": zx.rules.match_pivot_gadget,
            #     "apply": zx.rules.pivot_gadget,
            # },
            # "gadget_fusion": {
            #     "name": "gadget_fusion",
            #     "match": zx.rules.match_gadget_fusion,
            #     "apply": zx.rules.gadget_fusion,
            # },
        }

        database = []
        queue = [(self.g,0,"",[self.score(self.g)])]
        count = 0
        ids_hist = Histogram("ids")
        lcomps_hist = Histogram("lcomps")
        pivots_hist = Histogram("pivots")
        depthwise_hist = {}
        gates_hist = Histogram("gates")
        visited = dict()
        while len(queue) != 0:
            g1, depth, history, score_history = queue.pop(-1)
            current_score = score_history[-1]
            count += 1

            # 大きな回路でのidentity checkはできません
            # if not self.check_identity(g1):
            #     print("Identity check: ", self.check_identity(g1), history)

            if not depth in depthwise_hist:
                depthwise_hist[depth] = Histogram("depth " + str(depth))
                depthwise_hist[depth].add(current_score)
            else:
                depthwise_hist[depth].add(current_score)

            graph_hash = nx.weisfeiler_lehman_graph_hash(self.get_nx_graph(g1))
            if graph_hash in visited:
                self.score_histories.append(score_history)
                database.append(g1)
                continue
            else:
                # zx.draw_matplotlib(g1,labels=True,h_edge_draw='box').savefig("current.png")
                visited[graph_hash] = g1
                
            print("Queue length:", len(queue), history, current_score, )

            branch = 0
            for action in self.actions:
                for match in self.actions[action]["match"](g1):
                    branch += 1
                    g2 = deepcopy(g1)
                    etab, rem_verts, rem_edges, check_isolated_vertices = self.actions[action]["apply"](g2, [match])
                    try:
                        before = g2.num_vertices()
                        g2.add_edge_table(etab)
                        g2.remove_edges(rem_edges)
                        g2.remove_vertices(rem_verts)
                        if check_isolated_vertices:
                            g2.remove_isolated_vertices()

                        new_score = self.score(g2)
                        if self.should_explore(depth, new_score, current_score):
                            queue.append((g2,depth+1,history + f" {self.actions[action]['name']}({new_score - current_score})", score_history + [new_score]))
                        after = g2.num_vertices()
                        #print("id ", before, after)
                        if self.actions[action]["name"] == "id":
                            scatter_id.add(len(g1.neighbors(match[0])), new_score-current_score)
                        elif self.actions[action]["name"] == "lcomp":
                            scatter_lcomp.add(len(g1.neighbors(match[0])), new_score-current_score)
                        elif self.actions[action]["name"] == "pivot":
                            scatter_pivot.add(len(g1.neighbors(match[0])), new_score-current_score)

                        gates_hist.add(new_score)
                        if  new_score <= self.min_score:
                            self.min_score = new_score
                            min_history = history + f" {self.actions[action]['name']}({new_score - current_score})"
                            self.min_g = g2
                            # print(self.min_score, min_history)
                    except Exception as e:
                        print(e)

            if branch == 0:
                self.score_histories.append(score_history)
                database.append(g1)


        # databases = []
        # try:
        #     left = pickle.load(open("database.pkl", "rb"))
        #     databases.extend(left)
        # except Exception as e:
        #     print(e)
        # databases.append(database)
        # pickle.dump(databases, open("database.pkl", "wb"))
        
        print("Min score: ", self.min_score)
        print("Min history: ", min_history)
        print("Traversal count: ", count)

    def should_explore(self, depth, new_score, current_score):
        explore_prob = 0.01
        # return depth < 3 or new_score < current_score
        return True

    # less is better
    def score(self, _g):
        # circuit = zx.extract.extract_clifford_normal_form(_g.copy()).to_basic_gates()
        circuit = zx.extract.extract_circuit(_g.copy()).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        return len(circuit.gates)
        def get_data(circuit):
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

        def obtain_gates_pyzx(g):
            graph = g.copy()
            zx.simplify.to_graph_like(graph)
            #zx.simplify.flow_2Q_simp(graph)
            #zx.simplify.flow_2Q_simp(graph)
            circuit = zx.extract.extract_clifford_normal_form(graph).to_basic_gates()

            circuit = zx.basic_optimization(circuit).to_basic_gates()
            return get_data(circuit)

class RiuOptimizer1(TraversalOptimizer):
    def __init__(self, circ):
        super().__init__(circ)
        self.device = torch.device("cuda")
        self.agent = rl_agent.AgentGNN(self.device).to(self.device)
        self.agent.load_state_dict(
            torch.load("state_dict_model5x60_new.pt", map_location=torch.device("cpu"))
        )
        self.agent.eval()  
                #  Action: id, pivot, pivot_boudary, pivot_gadget, lcomp, gadget fusion

    def run(self):
        min_depth = 10000
        self.min_score = 10000
        self.min_g = None
        min_history = ""
        self.score_histories = []

        self.rl_ctx = zx_env.RLContext()

        database = []
        initial_score= self.score(self.g)
        queue = [(self.g,0,"",[initial_score])]
        count = 0
        ids_hist = Histogram("ids")
        lcomps_hist = Histogram("lcomps")
        pivots_hist = Histogram("pivots")
        depthwise_hist = {}
        gates_hist = Histogram("gates")
        visited = dict()
        timer = Timer(silent=False)


        while len(queue) != 0:
            timer.cut("loop begin", silent=False)
            g1, depth, history, score_history = queue.pop(0)
            current_score = score_history[-1]
            count += 1

            print(current_score, history)

            #print( self.check_identity(g1), history,)

            # if not self.check_identity(g1):
            #     print("Identity check: ", self.check_identity(g1), history)

            if not depth in depthwise_hist:
                depthwise_hist[depth] = Histogram("depth " + str(depth))
                depthwise_hist[depth].add(current_score)
            else:
                depthwise_hist[depth].add(current_score)

            graph_hash = nx.weisfeiler_lehman_graph_hash(self.get_nx_graph(g1))
            if graph_hash in visited:
                self.score_histories.append(score_history)
                database.append(g1)
                continue
            else:
                # zx.draw_matplotlib(g1,labels=True,h_edge_draw='box').savefig("current.png")
                visited[graph_hash] = g1
                
            branch = 0
            self.rl_ctx.update_state(g1)
            before = g1.num_vertices()
            actions = self.rl_ctx.policy_obs(g1)

            # drop random any% actions
            actions = [act for act in actions if dice(0.2)]
            print("Traverse actions: ", len(actions))
            time_lambda(self, "Scoreing time", lambda: self.score(self.g))

            for act in actions:
                timer.cut("action loop", silent=True)
                branch += 1
                g2 = deepcopy(g1)
                g2, act_str = self.rl_ctx.step(g2, act)
                count += 1
                after = g2.num_vertices()
                #print(act_str, after-before)
                if act_str == "ID":
                    continue
                if act_str == "STOP":
                    continue
                timer.cut("action loop 2", silent=True)
                new_score = self.score(g2)
                timer.cut("action loop 3", silent=True)
                if self.should_explore(depth, new_score, current_score):
                    queue.append((g2,depth+1,history + f" {act_str}({new_score - current_score})", score_history + [new_score]))
                if new_score < self.min_score:
                    self.min_score = new_score
                    min_history = history + f" {act_str}({new_score - current_score})"
                    # print("*", self.min_score, self.min_score-initial_score, min_history)
                    self.min_g = g2

            if branch == 0:
                self.score_histories.append(score_history)
                database.append(g1)


            timer.cut("action loop end", silent=True)
      # databases = []
        # try:
        #     left = pickle.load(open("database.pkl", "rb"))
        #     databases.extend(left)
        # except Exception as e:
        #     print(e)
        # databases.append(database)
        # pickle.dump(databases, open("database.pkl", "wb"))
        
        print("Min score: ", self.min_score-initial_score)
        print("Min history: ", min_history)
        print("Traversal count: ", count)

    def should_explore(self, depth, new_score, current_score):
        return new_score < current_score
    
    # less is better
    def score(self, _g):
        # circuit = zx.extract.extract_clifford_normal_form(_g.copy()).to_basic_gates()
        circuit = zx.extract.extract_circuit(_g.copy()).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        return len(circuit.gates)
    
        value_obs = self.rl_ctx.value_obs(_g)
        return -self.agent.get_value(value_obs).item()

    def lc_heuristics(self, _g, u):
        pass

    def pivot_heuristics(self, _g, u, v):
        pass

sys.path.append("./RL/")
sys.path.append("./zx/RL/")
import gymnasium as gym
from RL.src.training_method.ppo import PPO
from RL.src.agenv.zxopt_agent import AgentGNN
from RL.src.util import CustomizedAsyncVectorEnv, CustomizedSyncVectorEnv

# 強化学習エージェントの出力を参考に探索する
class RLOptimizer(TraversalOptimizer):

    def __init__(self, circ):
        super().__init__(circ)
        self.device = torch.device("cuda")
        def make_env(gym_id, seed, idx, capture_video, run_name, gate_type, g):
            def thunk():
                env = gym.make(gym_id, gate_type = gate_type, g = g)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                if capture_video and idx == 0:
                    env = gym.wrappers.RecordVideo(env, rootdir(f"/videos/{run_name}"))
                # env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        self.agent = get_agent_from_state_dict(None, self.device, self.args, torch.load("/home/wsl/Research/nogami/zx/state_dict_model5x60_new.pt", map_location=torch.device("cpu")).to(self.device)
        self.agent.eval()

                #  Action: id, pivot, pivot_boudary, pivot_gadget, lcomp, gadget fusion

    def run(self):
        min_depth = 10000
        self.min_score = 10000
        self.min_g = None
        min_history = ""
        self.score_histories = []

        self.rl_ctx = zx_env.RLContext()

        database = []
        initial_score= self.score(self.g)
        queue = [(self.g,0,"",[initial_score])]
        count = 0
        ids_hist = Histogram("ids")
        lcomps_hist = Histogram("lcomps")
        pivots_hist = Histogram("pivots")
        depthwise_hist = {}
        gates_hist = Histogram("gates")
        visited = dict()
        timer = Timer(silent=False)


        while len(queue) != 0:
            timer.cut("loop begin", silent=False)
            g1, depth, history, score_history = queue.pop(0)
            current_score = score_history[-1]
            count += 1

            print(current_score, history)

            #print( self.check_identity(g1), history,)

            # if not self.check_identity(g1):
            #     print("Identity check: ", self.check_identity(g1), history)

            # if not depth in depthwise_hist:
            #     depthwise_hist[depth] = Histogram("depth " + str(depth))
            #     depthwise_hist[depth].add(current_score)
            # else:
            #     depthwise_hist[depth].add(current_score)

            # graph_hash = nx.weisfeiler_lehman_graph_hash(self.get_nx_graph(g1))
            # if graph_hash in visited:
            #     self.score_histories.append(score_history)
            #     database.append(g1)
            #     continue
            # else:
            #     # zx.draw_matplotlib(g1,labels=True,h_edge_draw='box').savefig("current.png")
            #     visited[graph_hash] = g1
            before = g1.num_vertices()
                
            branch = 0
            self.rl_ctx.update_state(g1)
            info = self.rl_ctx.get_info(g1)
            policy_feat = self.agent.get_policy_feature_graph(g1, info)
            logits = self.agent.actor(policy_feat).flatten()
            mask = policy_feat.y != -1
            logits = logits[mask]
            actions = policy_feat.y[mask]
            logit_action = zip(logits, actions)
            logit_action = sorted(logit_action, key=lambda x: x[0], reverse=True)
            actions = [x[1] for x in logit_action if x[0] > -10]
            # actions = [x[1] for x in logit_action]

            # drop random any% actions
            # actions = [act for act in actions if dice(0.2)]
            print("Traverse actions: ", len(actions))
            time_lambda(self, "Scoreing time", lambda: self.score(self.g))

            for act in actions:
                timer.cut("action loop", silent=True)
                branch += 1
                count += 1
                g2 = deepcopy(g1)
                g2, act_str = self.rl_ctx.step(g2, act)
                after = g2.num_vertices()
                #print(act_str, after-before)
                # if act_str == "ID":
                #     continue
                if act_str == "STOP":
                    continue
                timer.cut("action loop 2", silent=True)
                new_score = self.score(g2)
                timer.cut("action loop 3", silent=True)
                # if self.should_explore(depth, new_score, current_score):
                queue.append((g2,depth+1,history + f" {act_str}({new_score - current_score})", score_history + [new_score]))
                if new_score < self.min_score:
                    self.min_score = new_score
                    min_history = history + f" {act_str}({new_score - current_score})"
                    # print("*", self.min_score, self.min_score-initial_score, min_history)
                    self.min_g = g2

            if branch == 0:
                self.score_histories.append(score_history)
                database.append(g1)


            timer.cut("action loop end", silent=True)
      # databases = []
        # try:
        #     left = pickle.load(open("database.pkl", "rb"))
        #     databases.extend(left)
        # except Exception as e:
        #     print(e)
        # databases.append(database)
        # pickle.dump(databases, open("database.pkl", "wb"))
        
        print("Min score: ", self.min_score-initial_score)
        print("Min history: ", min_history)
        print("Traversal count: ", count)

    def should_explore(self, depth, new_score, current_score):
        return new_score < current_score
    
    # less is better
    def score(self, _g):
        # circuit = zx.extract.extract_clifford_normal_form(_g.copy()).to_basic_gates()
        circuit = zx.extract.extract_circuit(_g.copy()).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        return len(circuit.gates)
    
        value_obs = self.rl_ctx.value_obs(_g)
        return -self.agent.get_value(value_obs).item()

    def lc_heuristics(self, _g, u):
        pass

    def pivot_heuristics(self, _g, u, v):
        pass

class RiuOptimizer_WithPolicy(TraversalOptimizer):
    static_initialized = False
    device = torch.device("cuda")
    agent = rl_agent.AgentGNN(device).to(device)

    def __init__(self, circ, search_policy, initial_queue=None):
        # 静的変数の初期化
        if not RiuOptimizer_WithPolicy.static_initialized:
            RiuOptimizer_WithPolicy.agent.load_state_dict(
                torch.load("zx/state_dict_model5x60_new.pt", map_location=torch.device("cpu"))
            )
            RiuOptimizer_WithPolicy.static_initialized = True

        if circ is not None:
            super().__init__(circ)
        else:
            self.g = None
        self.initial_queue = initial_queue

        print(" policy = ", search_policy)


        self.search_policy = search_policy.split("\n")
        self.search_policy = [stmt for stmt in self.search_policy if stmt != ""]
        self.program_counter = 0
        self.current_search = self.parse_parameter(self.search_policy[self.program_counter])
        self.current_condition = self.parse_parameter(self.search_policy[self.program_counter+1])
        RiuOptimizer_WithPolicy.agent.eval()  
                #  Action: id, pivot, pivot_boudary, pivot_gadget, lcomp, gadget fusion
    
    ### 現時点でのキューの中からスコアが最も高いものk個のみを残す
    def COMMAND_topk(self, k):
        dict_prev = {}
        for e in self.queue:
            g1, depth, history, score_history = e
            if not score_history[-1] in dict_prev:
                dict_prev[score_history[-1]] = []
            dict_prev[score_history[-1]].append(e)
        self.queue = []

        dict_prev = sorted(dict_prev.items(), key=lambda x: x[0])
        sorted_prev  = []
        for k, v in dict_prev:
            for e in v:
                sorted_prev.append(e)
        try:
            for i in range(int(self.current_search['k'])):
                self.queue.append(sorted_prev[i])
        except:
            pass
        return 
    
    ### ゲートが減っているという条件
    ### num_paths : 何本ゲートが減るパスが見つかると条件を満たすか(デフォルト1)
    ### reduction : ゲートが減っているとみなすゲート減少量(デフォルト-1)
    def CONDITION_GateReduction(self, num_paths, reduction):
        if num_paths == None: num_paths = 1
        if reduction == None: reduction = -1
        count = 0
        for q in self.queue:
            g, depth, history, score_history = q
            if score_history[-1] - self.initial_score <= reduction:
                count += 1
            if count >= num_paths:
                return True
        return False


    def SEARCH_TopKHolding(self, dict_next):
        dict_prev = {}
        for e in self.queue:
            g1, depth, history, score_history = e
            if not score_history[-1] in dict_prev:
                dict_prev[score_history[-1]] = []
            dict_prev[score_history[-1]].append(e)
        queue = []

        dict_prev = sorted(dict_prev.items(), key=lambda x: x[0])
        dict_next = sorted(dict_next.items(), key=lambda x: x[0])
        sorted_next  = []
        for k, v in dict_next:
            for e in v:
                sorted_next.append(e)
        sorted_prev  = []
        for k, v in dict_prev:
            for e in v:
                sorted_prev.append(e)

        queue.append(sorted_next[0])
        queue.append(sorted_next[1])
        for i in range(int(self.current_search['k'])):
            queue.append(sorted_prev[i])


    def SEARCH_TopKSelection(self, dict_next):
        dict_next = sorted(dict_next.items(), key=lambda x: x[0])
        sorted_next  = []
        for k, v in dict_next:
            for e in v:
                sorted_next.append(e)
        
        for i in range(int(self.current_search['k'])):
            self.queue.append(sorted_next[i])


    def SEARCH_GreedySearch(self, dict_next):
        dict_next = sorted(dict_next.items(), key=lambda x: x[0])
        sorted_next  = []
        for k, v in dict_next:
            for e in v:
                sorted_next.append(e)
        self.queue.append(sorted_next[0])


    def SEARCH_FullSearch(self, dict_next):
        dict_next = sorted(dict_next.items(), key=lambda x: x[0])
        sorted_next  = []
        for k, v in dict_next:
            for e in v:
                sorted_next.append(e)
        for i in range(len(sorted_next)):
            self.queue.append(sorted_next[i])


    def parse_parameter(self, stmt):
        stmt = stmt.strip()
        stmt = stmt.split(' ')
        result = {}
        wptr = 0
        if stmt[0] == 'until':
            wptr += 1
        result['op'] = stmt[wptr] ; wptr += 1
        for i in range(wptr, len(stmt)):
            try:
                key, value = stmt[i].split('=')
                try:
                    result[key] = float(value)
                except ValueError as e:
                    result[key] = str(value)
            except:
                result[stmt[i]] = None
        return result

    def next_program_counter(self, lines=2):
        self.program_counter += lines
        if self.program_counter >= len(self.search_policy):
            return False
        
        try:
            self.current_search = self.parse_parameter(self.search_policy[self.program_counter])
            self.current_condition = self.parse_parameter(self.search_policy[self.program_counter+1])
        except:
            return False
        
        # 即時実行コマンドがあれば、実行して一行進める
        if self.current_search['op'][0] == '!':
            if self.current_search['op'] == '!topk':
                self.COMMAND_topk(k=int(self.current_search['k']))
            self.next_program_counter(lines=1)
            return True
        
        # separate操作があれば、実行する
        if self.current_search['op'] == 'separate':
            separate_search_policy = self.search_policy[self.program_counter:]
            separate_search_policy[0] = " ".join(separate_search_policy[0].split(" ")[1:])
            for q in self.queue:
                opt = RiuOptimizer_WithPolicy(None, "\n".join(separate_search_policy), initial_queue=[q])
                opt.run()
                if opt.min_score < self.min_score:
                    self.min_score = opt.min_score
                    self.min_g = opt.min_g
                    self.min_history = opt.min_history
            return True
        return True
    
    def init(self):
        import time
        self.min_score = 10000
        self.min_g = None
        self.min_history = ""
        self.score_histories = []

        self.rl_ctx = zx_env.RLContext()

        if self.initial_queue is not None:
            self.queue = self.initial_queue
            # NOTE(malick): とりあえず、最初のスコアを取得しておく
            _, _, _, score_history = self.queue[0]
            self.initial_score = score_history[-1]
        else:
            self.initial_score= self.score(self.g)
            self.queue = [(self.g,0,"",[self.initial_score])]
            
        
        self.count = 0
        self.database = []
        # ids_hist = Histogram("ids")
        # lcomps_hist = Histogram("lcomps")
        # pivots_hist = Histogram("pivots")
        # depthwise_hist = {}
        # gates_hist = Histogram("gates")
        self.visited = dict()
        self.timer = Timer(silent=False)

        self.search_data = {
            'no_fruit_count': 0
        }

    
    def run(self):
        self.init()
        start_time = time.time()

        while len(self.queue) != 0:
            if not self.run_once():
                break
            # 実行が300秒を超えたら終了            
            if time.time() > start_time + 300:
                break

        # databases = []
        # try:
        #     left = pickle.load(open("database.pkl", "rb"))
        #     databases.extend(left)
        # except Exception as e:
        #     print(e)
        # databases.append(database)
        # pickle.dump(databases, open("database.pkl", "wb"))
        
        print("Min score: ", self.min_score-self.initial_score)
        print("Min history: ", self.min_history)
        print("Traversal count: ", self.count)


    # 一回探索する
    def run_once(self):
        self.timer.cut("loop begin", silent=False)
        g1, depth, history, score_history = self.queue.pop(0)
        current_score = score_history[-1]
        self.count += 1

        print(current_score, history)

        #print( self.check_identity(g1), history,)

        # if not self.check_identity(g1):
        #     print("Identity check: ", self.check_identity(g1), history)

        # if not depth in depthwise_hist:
        #     depthwise_hist[depth] = Histogram("depth " + str(depth))
        #     depthwise_hist[depth].add(current_score)
        # else:
        #     depthwise_hist[depth].add(current_score)

        graph_hash = nx.weisfeiler_lehman_graph_hash(self.get_nx_graph(g1))
        if graph_hash in self.visited:
            self.score_histories.append(score_history)
            self.database.append(g1)
            return True
        else:
            # zx.draw_matplotlib(g1,labels=True,h_edge_draw='box').savefig("current.png")
            self.visited[graph_hash] = g1
            
        branch = 0
        self.rl_ctx.update_state(g1)
        before = g1.num_vertices()
        actions = self.rl_ctx.policy_obs(g1)
        no_fruit = True
        
        dict_next = {}


        # drop random any% actions
        if 'p' in self.current_search:
            actions = [act for act in actions if dice(self.current_search['p'])]
        
        print(self.current_search['op'] + " on actions: ", len(actions))
        time_lambda(self, "Scoreing time", lambda: self.score(g1))

        for act in actions:
            self.timer.cut("action loop", silent=True)
            branch += 1
            g2 = deepcopy(g1)
            g2, act_str = self.rl_ctx.step(g2, act)
            after = g2.num_vertices()
            #print(act_str, after-before)
            if act_str == "ID":
                continue
            if act_str == "STOP":
                continue
            self.timer.cut("action loop 2", silent=True)
            new_score = self.score(g2)
            self.timer.cut("action loop 3", silent=True)
            if not new_score in dict_next:# and new_score <= current_score:
                dict_next[new_score] = []
                dict_next[new_score].append((g2,depth+1,history + f" {act_str}({new_score - current_score})", score_history + [new_score]))
            # if self.should_explore(depth, new_score, current_score):
            #     queue.append((g2,depth+1,history + f" {act_str}({new_score - current_score})", score_history + [new_score]))
            if new_score < self.min_score:
                self.min_score = new_score
                self.min_history = history + f" {act_str}({new_score - current_score})"
                no_fruit = False
                print("*", self.min_score, self.min_score-self.initial_score, self.min_history)
                self.min_g = g2

        # 探索するものを追加
        try:
            if self.current_search['op'] == 'TopKHoldingSearch':
                self.SEARCH_TopKHolding(dict_next)
            elif self.current_search['op'] == 'TopKSelectionSearch':
                self.SEARCH_TopKSelection(dict_next)
            elif self.current_search['op'] == 'GreedySearch':
                self.SEARCH_GreedySearch(dict_next)
            elif self.current_search['op'] == 'FullSearch':
                self.SEARCH_FullSearch(dict_next)
        except Exception as e:
            pass


        # program_counterを進めるかどうかをチェック
        if self.current_condition['op'] == 'Nofruit':
            if no_fruit:
                self.search_data['no_fruit_count'] += 1
            else:
                self.search_data['no_fruit_count'] = 0

            if self.search_data['no_fruit_count'] >= int(self.current_condition.get('wait')):
                if not self.next_program_counter():
                    return False
        
        if self.current_condition['op'] == 'GateReduction' and \
            self.CONDITION_GateReduction(self.current_condition.get('num_paths'), self.current_condition.get('reduction')):
            if not self.next_program_counter():
                return False


        if branch == 0:
            self.score_histories.append(score_history)
            self.database.append(g1)
        
        if 'shuffle' in self.current_search and self.current_search['shuffle'] == 'True':
            # shuffle queue
            random.shuffle(self.queue)

        self.timer.cut("action loop end", silent=True)
        return True

    def should_explore(self, depth, new_score, current_score):
        return new_score < current_score
    
    # less is better
    def score(self, _g):
        # circuit = zx.extract.extract_clifford_normal_form(_g.copy()).to_basic_gates()
        circuit = zx.extract.extract_circuit(_g.copy()).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        return len(circuit.gates)
    
        value_obs = self.rl_ctx.value_obs(_g)
        return -RiuOptimizer_WithPolicy.agent.get_value(value_obs).item()
    
    def lc_heuristics(self, _g, u):
        pass

    def pivot_heuristics(self, _g, u, v):
        pass

def divide_circuit(g, divide):
    qubits = g.qubit_count()
    _inputs = g.inputs()
    lines = [[e] for e in _inputs]
    visited = set(_inputs)
    print(lines)
    finished = False
    
    # まず横のラインをとる
    while not finished:
        finished = True
        for i in range(qubits):
            neigh = g.neighbors(lines[i][-1])
            neigh = [e for e in neigh if e not in visited]
            if len(neigh) == 1:
                visited.add(neigh[0])
                lines[i].append(neigh[0])
                finished = False
            elif len(neigh) > 1:
                pass
            else:
                pass
    
    # 等間隔の分割を目指すが、縦のラインを飛び越えて分割はできないので調整しながらやる
    desired_length = [len(line)//divide for line in lines]
    actual_lengths = [[] for _ in lines]

    print(lines)


def score_heavy(g):
    circuit = zx.Circuit.from_graph(_g_circ).split_phase_gates()
    circuit = zx.basic_optimization(circuit).to_basic_gates()
    return len(circuit.gates)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RL optimizer')
    parser.add_argument('--qasm', type=str, default=None, help='circuit file')
    parser.add_argument('--cut-logit', type=int, default=None, help='circuit file')
    args = parser.parse_args()

    g_circ = zx.Circuit.load(args.qasm).to_basic_gates().to_graph()
    opt1 = RLOptimizer(g_circ)
    opt1.run()


    program = """
    FullSearch
    until GateReduction num_paths=5
    !topk k=2
    separate GreedySearch p=0.5
    until Nofruit wait=10
    """

    program2 = """
    TopKSelectionSearch k=2 p=1.0
    until Nofruit wait=5
    """

    results = {
        'overall': [],
        'fromBO': [],
    }
    for i in range(100):
        # random.seed(i)
        g_circ = zx.generate.cliffordT(40,500)
        original_tensor = g_circ.to_tensor()

        # calculate crude gates
        _g_circ = g_circ.copy()
        # print(len(zx.extract_circuit(_g_circ).gates))
        #gates_no_opt = _g_circ.num_vertices() - 4*_g_circ.qubit_count()
        c = zx.Circuit.from_graph(_g_circ)
        gates_no_opt = len(c.to_basic_gates().gates)


        # divide_circuit(g_circ, 10)
        opt1 = RLOptimizer(g_circ)
        # opt1 = RiuOptimizer_WithPolicy(g_circ, program2)
        opt1.run()

        circuit = zx.extract.extract_circuit(opt1.min_g.copy()).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        # print("no opt", gates_no_opt)
        # print("bo", score_heavy(_g_circ))
        # print("Min: ", len(circuit.gates), len(circuit.gates) - gates_no_opt)
        circuit_tensor = circuit.to_tensor()
        if zx.compare_tensors(original_tensor, circuit_tensor):
            print("OK")
        else:
            print("NG")

        results['fromBO'].append(opt1.min_score - opt1.initial_score)
        results["overall"].append(opt1.min_score - gates_no_opt)
        # print()

    print("max", np.max(results['fromBO']), "min", np.min(results['fromBO']), "mean", np.mean(results['fromBO']), "std", np.std(results['fromBO']))
    print("max", np.max(results['overall']), "min", np.min(results['overall']), "mean", np.mean(results['overall']), "std", np.std(results['overall']))

    # scatter_id.save()
    # scatter_lcomp.save()
    # scatter_pivot.save()

    scatter_id.print()
    plt.show()
    scatter_lcomp.print()
    plt.show()
    scatter_pivot.print()
    plt.show()