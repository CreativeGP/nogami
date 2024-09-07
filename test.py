import pyzx as zx
import random
from copy import deepcopy

matchf = None
quiet = True
stats = None
error_list = []
for i in range(100):
    qubit = 5
    depth = 10
    random.seed(i)
    g = zx.generate.cliffordT(qubit, depth)
    g1 = deepcopy(g)
    # inte
    # zx.simplify.spider_simp(g1, quiet=True)
    zx.simplify.to_gh(g1)
    while True:
        i1 = zx.id_simp(g1, matchf=matchf, quiet=quiet, stats=stats)
        # i2 = zx.spider_simp(g1, matchf=matchf, quiet=quiet, stats=stats)
        i3 = zx.pivot_simp(g1, matchf=matchf, quiet=quiet, stats=stats)
        i4 = zx.lcomp_simp(g1, matchf=matchf, quiet=quiet, stats=stats)
        if  i1 + i3 + i4 ==0: break
    zx.simplify.to_graph_like(g1)
    circ = zx.extract_circuit(g1)
    t1 = g.to_tensor()
    t2 = circ.to_tensor()
    if zx.compare_tensors(t1, t2):
        pass
    else:
        print(False)
        error_list.append(g)

if error_list == []:
    print("all correct")
else:
    print("error found")