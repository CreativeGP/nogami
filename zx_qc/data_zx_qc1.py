import random
import pyzx as zx
from copy import deepcopy
from collections import Counter
from fractions import Fraction
import pandas as pd
import time

num_qubits = 10
num_depth = 50

num_sampling = 1000
#zx-diagram data collection
tmp_dg_all = []
tmp_qc_all = []
tmp_qc1_all = []

for i in range(num_sampling):
    start_time = time.time()
    random.seed(i)
    diagram = zx.generate.cliffordT(num_qubits, num_depth)
    zx.simplify.full_reduce(diagram)
    diagram.normalize()
    copy = deepcopy(diagram)
    qc = zx.extract_circuit(copy)
    qc1 = zx.full_optimize(qc)

    # datacollection Diagram
    #print(diagram.num_edges(), diagram.num_vertices())
    num_e = diagram.num_edges()-2*num_qubits
    #print(num_e)
    num_v = diagram.num_vertices()-2*num_qubits
    #print(num_v)
    tmp_edge = diagram.edge_set()
    flat = [item for sublist in tmp_edge for item in sublist]
    ct = Counter(flat)
    ct1 = {key: count for key, count in ct.items() if count >= 3}
    tmp1 = sum(ct1.values())
    tmp2 = 2 * len(ct1)
    complicated_index = len([i for i in ct1.values() if i>3])
    num_verticallyconnected_edges = int((tmp1-tmp2)/2)
    tmp_phases = diagram.phases()
    tmp_phases1 = [float(2*value) for value in tmp_phases.values()]
    noncliffordphases = [Fraction(x/2) for x in tmp_phases1 if int(x*10)%10==5]
    num_noncliffordsphase_vertices = len(noncliffordphases)
    num_edge_and_vertices = num_e + num_v

    # organize
    tmp_dg = [num_e, num_v, num_verticallyconnected_edges, num_noncliffordsphase_vertices, num_edge_and_vertices, complicated_index]
    #print(tmp_dg)
    tmp_dg_all.append(tmp_dg)

    #datacollection qc
    num_qc_gates = qc.stats_dict()["gates"]
    num_qc_2qgates = qc.stats_dict()["twoqubit"]
    num_qc_noncliffordsgates = qc.stats_dict()["tcount"]
    num_qc_hadamard = qc.stats_dict()["had"]

    # organaize
    tmp_qc = [num_qc_gates, num_qc_2qgates, num_qc_noncliffordsgates, num_qc_hadamard]
    tmp_qc_all.append(tmp_qc)

    #datacollenction qc1
    num_qc1_gates = qc1.stats_dict()["gates"]
    num_qc1_2qgates = qc1.stats_dict()["twoqubit"]
    num_qc1_noncliffordsgates = qc1.stats_dict()["tcount"]
    num_qc1_hadamard = qc1.stats_dict()["had"]

    # organize
    tmp_qc1 = [num_qc1_gates, num_qc1_2qgates, num_qc1_noncliffordsgates, num_qc1_hadamard]
    tmp_qc1_all.append(tmp_qc1)

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"{i}番目の実行時間:{execution_time}")

columns_diagram_list = ["num_edge", "num_vertices", "num_verticallyconnected_edges", "num_noncliffordsphase_vertices", "num_edge_and_vertices", "complicated_index"]
df_dg = pd.DataFrame(tmp_dg_all, columns=columns_diagram_list)

columns_qc_list = ["num_gates", "num_2qgates", "num_noncliffordsgates", "num_hadamardgates"]
df_qc = pd.DataFrame(tmp_qc_all, columns=columns_qc_list)
df_qc1 = pd.DataFrame(tmp_qc1_all, columns=columns_qc_list)
print("データが完成しました。")

# store all data
save_dir = "/home/hinog/src/python3/pyzx_utils/zx_qc/data/pkl1/"
df_dg.to_pickle(f"{save_dir}{num_qubits}_{num_depth}_zxdiagram.pkl")
df_qc.to_pickle(f"{save_dir}{num_qubits}_{num_depth}_qc.pkl")
df_qc1.to_pickle(f"{save_dir}{num_qubits}_{num_depth}_qc1.pkl")