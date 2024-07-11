import sys; sys.path.insert(0, '../..') # So that we import the local copy of pyzx if you have installed from Github
import random
import pyzx as zx
from copy import deepcopy
import json
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
import networkx as nx


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

def dice(p):
    return random.random() < p

class Optimizer(ABC):
    def __init__(self, circ):
        self.g = deepcopy(circ)
        self.g_circ = deepcopy(circ)
        # zx.simplify.spider_simp(self.g)
        zx.simplify.to_gh(self.g)

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

        self.actions = {
            "id": {
                "name": "id",
                "match": zx.rules.match_ids_parallel,
                "apply": zx.rules.remove_ids,
            },
            "lcomp": {
                "name": "lcomp",
                "match": zx.rules.match_lcomp_parallel,
                "apply": zx.rules.lcomp,
            },
            "pivot": {
                "name": "pivot",
                "match": zx.rules.match_pivot_parallel,
                "apply": zx.rules.pivot,
            }
        }

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

            # if not self.check_identity(g1):
            #     print("Identity check failed: ", history)


            # ids = zx.rules.match_ids_parallel(g1) # (identity_vertex, neighbor1, neighbor2, edge_type)
            # lcomps = zx.rules.match_lcomp_parallel(g1) # (V,V)
            # pivots = zx.rules.match_pivot_parallel(g1)# + zx.rules.match_pivot_boundary(g1) + zx.rules.match_pivot_gadget(g1) # ((V,V), (V,V))
            # print("ids", len(ids))
            # print("lcomps", len(lcomps))
            # print("pivots", len(pivots))
            # ids_hist.add(len(ids))
            # lcomps_hist.add(len(lcomps))
            # pivots_hist.add(len(pivots))
            if not depth in depthwise_hist:
                depthwise_hist[depth] = Histogram("depth " + str(depth))
                depthwise_hist[depth].add(current_score)
            else:
                depthwise_hist[depth].add(current_score)

            graph_hash = nx.weisfeiler_lehman_graph_hash(self.get_nx_graph(g1))
            if graph_hash in visited:
                self.score_histories.append(score_history)
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

                        # _g2 = deepcopy(g2)
                        # circ = zx.extract_circuit(_g2)
                        # circ = zx.basic_optimization(circ)
                        # if circ.depth() < min_depth:
                        #     min_depth = circ.depth()
                        #     print(min_depth)
                        gates_hist.add(new_score)
                        if  new_score <= self.min_score:
                            self.min_score = new_score
                            min_history = history + f" {self.actions[action]['name']}({new_score - current_score})"
                            self.min_g = g2
                            print(self.min_score, min_history)
                    except Exception as e:
                        print(e)

            if branch == 0:
                self.score_histories.append(score_history)



        
        print("Min score: ", self.min_score)
        print("Min history: ", min_history)
        print("Traversal count: ", count)

        self.save_image()
    
    def check_identity(self, argg):
        _argg = argg.copy()
        zx.simplify.to_graph_like(_argg)
        #zx.simplify.flow_2Q_simp(graph)
        #zx.simplify.flow_2Q_simp(graph)
        circuit = zx.extract.extract_clifford_normal_form(_argg).to_basic_gates()
        circuit = zx.basic_optimization(circuit).to_basic_gates()
        return zx.compare_tensors(circuit,self.g,preserve_scalar=False)

    def save_image(self):
        circ = zx.extract.extract_clifford_normal_form(self.min_g).to_basic_gates()
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


class Optimizer1(Optimizer):
    def should_explore(self, depth, new_score, current_score):
        explore_prob = 0.01
        # return depth < 3 or new_score < current_score
        return True

    # less is better
    def score(self, _g):
        circuit = zx.extract.extract_clifford_normal_form(_g.copy()).to_basic_gates()
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

while True:
    g_circ = zx.generate.cliffords(5,20)
    opt1 = Optimizer1(g_circ)
    opt1.run()
    import time
    time.sleep(1)
    print()
