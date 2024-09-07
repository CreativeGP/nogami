import pyzx as zx
import numpy as np

def bo(g):
    circuit = zx.Circuit.from_graph(g).split_phase_gates()
    circuit = zx.basic_optimization(circuit).to_basic_gates()
    return len(circuit.gates)


results = {
    'bo': [],
}
for i in range(40):
    # random.seed(i)
    g_circ = zx.generate.cliffordT(10,300)
    results['bo'].append(bo(g_circ))

for key in results:
    print(key, "max", np.max(results[key]), "min", np.min(results[key]), "mean", np.mean(results[key]), "std", np.std(results[key]))
