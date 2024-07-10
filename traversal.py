import sys; sys.path.insert(0, '../..') # So that we import the local copy of pyzx if you have installed from Github
import random
import pyzx as zx
from copy import deepcopy
import json
import numpy as np
import matplotlib.pyplot as plt

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
    circuit = zx.extract.extract_simple(graph).to_basic_gates()

    circuit = zx.basic_optimization(circuit).to_basic_gates()
    return get_data(circuit)

def score(_g):
        return obtain_gates_pyzx(_g)["gates"]

def check_identity(argg):
    _argg = argg.copy()
    zx.simplify.to_graph_like(_argg)
    #zx.simplify.flow_2Q_simp(graph)
    #zx.simplify.flow_2Q_simp(graph)
    circuit = zx.extract.extract_simple(_argg).to_basic_gates()
    circuit = zx.basic_optimization(circuit).to_basic_gates()
    return zx.compare_tensors(circuit,_g,preserve_scalar=False)

#random.seed(1337)

# Generate target graph
g_circ = zx.generate.cliffords(2,5)
g = deepcopy(g_circ)
zx.simplify.spider_simp(g)
zx.simplify.to_gh(g)

# Traversal
min_depth = 10000
min_gates = 10000
min_g = None
min_history = ""
score_histories = []

queue = [(g,0,"",[score(g)])]
count = 0
ids_hist = Histogram("ids")
lcomps_hist = Histogram("lcomps")
pivots_hist = Histogram("pivots")
depthwise_hist = {}
gates_hist = Histogram("gates")
while len(queue) != 0:
    g1, depth, history, score_history = queue.pop(-1)
    count += 1

    # if check_identity(g1):
    #     print("Identity check ok: ", history)

    #print("Queue length:", len(queue))

    ids = zx.rules.match_ids_parallel(g1) # (identity_vertex, neighbor1, neighbor2, edge_type)
    lcomps = zx.rules.match_lcomp_parallel(g1) # (V,V)
    pivots = zx.rules.match_pivot_parallel(g1)# + zx.rules.match_pivot_boundary(g1) + zx.rules.match_pivot_gadget(g1) # ((V,V), (V,V))
    # print("ids", len(ids))
    # print("lcomps", len(lcomps))
    # print("pivots", len(pivots))
    # ids_hist.add(len(ids))
    # lcomps_hist.add(len(lcomps))
    # pivots_hist.add(len(pivots))

    if not depth in depthwise_hist:
        depthwise_hist[depth] = Histogram("depth " + str(depth))
        depthwise_hist[depth].add(score(g1))
    else:
        depthwise_hist[depth].add(score(g1))

    if len(ids) + len(lcomps) + len(pivots) == 0:
        score_histories.append(score_history)

    for id_match in ids:
        g2 = deepcopy(g1)
        etab, rem_verts, rem_edges, check_isolated_vertices = zx.rules.remove_ids(g2, [id_match])
        try:
            before = g2.num_vertices()
            g2.add_edge_table(etab)
            g2.remove_edges(rem_edges)
            g2.remove_vertices(rem_verts)
            if check_isolated_vertices:
                g2.remove_isolated_vertices()
            queue.append((g2,depth+1,history + " id", score_history + [score(g2)]))
            after = g2.num_vertices()
            #print("id ", before, after)

            # _g2 = deepcopy(g2)
            # circ = zx.extract_circuit(_g2)
            # circ = zx.basic_optimization(circ)
            # if circ.depth() < min_depth:
            #     min_depth = circ.depth()
            #     print(min_depth)
            gates = obtain_gates_pyzx(g2)["gates"]
            gates_hist.add(gates)
            if  gates < min_gates:
                min_gates = gates
                min_history = history+" id"
                min_g = g2
                print(min_gates)
        except Exception as e:
            print(e)
    for pivot_match in pivots:
        g2 = deepcopy(g1)
        etab, rem_verts, rem_edges, check_isolated_vertices = zx.rules.pivot(g2, [pivot_match])
        try:
            before = g2.num_vertices()
            g2.add_edge_table(etab)
            g2.remove_edges(rem_edges)
            g2.remove_vertices(rem_verts)
            if check_isolated_vertices:
                g2.remove_isolated_vertices()
            queue.append((g2,depth+1, history + " pivot", score_history + [score(g2)]))
            after = g2.num_vertices()
           # print("pivot ", before, after)

            # _g2 = deepcopy(g2)
            # circ = zx.extract_circuit(_g2)
            # circ = zx.basic_optimization(circ)
            # if circ.depth() < min_depth:
            #     min_depth = circ.depth()
            #     print(min_depth)
            gates = obtain_gates_pyzx(g2)["gates"]
            gates_hist.add(gates)
            if  gates < min_gates:
                min_gates = gates
                min_history = history+" pivot"
                min_g = g2
                print(min_gates)

        except:
            pass
    for lcomp_match in lcomps:
        g2 = deepcopy(g1)
        etab, rem_verts, rem_edges, check_isolated_vertices = zx.rules.lcomp(g2, [lcomp_match])
                
        try:
            before = g2.num_vertices()
            g2.add_edge_table(etab)
            g2.remove_edges(rem_edges)
            g2.remove_vertices(rem_verts)
            if check_isolated_vertices:
                g2.remove_isolated_vertices()
            queue.append((g2,depth+1, history + " lcomp", score_history + [score(g2)]))
            after = g2.num_vertices()
            #print("lcomp ", before, after)

            # _g2 = deepcopy(g2)
            # circ = zx.extract_circuit(_g2)
            # circ = zx.basic_optimization(circ)
            # if circ.depth() < min_depth:
            #     min_depth = circ.depth()
            #     print(min_depth)
            gates = obtain_gates_pyzx(g2)["gates"]
            gates_hist.add(gates)
            if  gates < min_gates:
                min_history = history+" lcomp"
                min_gates = gates
                min_g = g2
                print(min_gates)

        except:
            pass



print(min_gates, count)
# gates_hist.print()
# for i in range(len(depthwise_hist)):
#     depthwise_hist[i].print()
# ids_hist.print()
# lcomps_hist.print()
# pivots_hist.print()

    # Show only fig3

def save_image():
    fig = zx.draw_matplotlib(g_circ,labels=True,h_edge_draw='box')
    # circ = zx.basic_optimization(g_circ).to_basic_gates()
    fig.savefig("before.png")
    circ = zx.extract.extract_simple(min_g).to_basic_gates()
    circ = zx.basic_optimization(circ)
    fig = zx.draw_matplotlib(circ,labels=True,h_edge_draw='box')
    fig.savefig("after.png")

    # concatenate images
    from PIL import Image
    im1 = Image.open("before.png")
    im2 = Image.open("after.png")
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.save("before-after.png")
    
    # remove before.png and after.png
    import os
    os.remove("before.png")
    os.remove("after.png")

def plot_traversal_graph():
    for i, history in enumerate(score_histories):
        x = list(range(1,len(history)+1))
        plt.plot(x, history, alpha=0.05)
    plt.show()
