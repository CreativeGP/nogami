from time import time
import pyzx as zx
from pyzx.simplify import full_reduce, flow_2Q_simp
from pyzx.generate import cliffordT
from .optimizer_logger import RandomCircuit_OptimizerLogger
import random
from copy import deepcopy

if __name__ == '__main__':
    num_tests = 1000
    qubits_gates = {}
    qubits_gates[5] = [50, 100, 150, 200]
    qubits_gates[10] = [100, 200, 300, 400]
    qubits_gates[20] = [200, 400, 600, 800]
    qubits_gates[40] = [400, 800, 1200, 1600]
    qubits_gates[80] = [800, 1600, 2400, 3200]
    qubits_gates[100] = [1000, 2000, 3000, 4000]
    for qubits, gate_list in qubits_gates.items():
        for gates in gate_list:
            logger1 = RandomCircuit_OptimizerLogger("pyzx_full_reduce", qubits, gates)
            logger2 = RandomCircuit_OptimizerLogger("pyzx_cflow", qubits, gates)
            for seed in range(num_tests):
                random.seed(seed)
                g = cliffordT(qubits, gates)
                original_circuit = zx.Circuit.from_graph(g).to_basic_gates()
                logger1.log_original_circuit_info(seed, original_circuit.gates, original_circuit.depth(), original_circuit.tcount(), original_circuit.twoqubitcount())
                logger2.log_original_circuit_info(seed, original_circuit.gates, original_circuit.depth(), original_circuit.tcount(), original_circuit.twoqubitcount())
                g_ = deepcopy(g)
                # full_reduce
                start = time()
                full_reduce(g_)
                optimized_c = zx.extract.extract_circuit(g_)
                optimized_c = zx.optimize.basic_optimization(optimized_c.to_basic_gates()).to_basic_gates()
                logger1.log_optimized_circuit_info(seed, optimized_c.gates, optimized_c.depth(), optimized_c.tcount(), optimized_c.twoqubitcount(), time() - start)
                
                g_ = deepcopy(g)
                # cflow
                start = time()
                flow_2Q_simp(g_)
                optimized_c = zx.extract.extract_simple(g_)
                optimized_c = zx.optimize.basic_optimization(optimized_c.to_basic_gates()).to_basic_gates()
                logger2.log_optimized_circuit_info(seed, optimized_c.gates, optimized_c.depth(), optimized_c.tcount(), optimized_c.twoqubitcount(), time() - start)
            
            logger1.finish_log()
                

