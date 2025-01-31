from time import time
import pyzx as zx
from pyzx.simplify import full_reduce, flow_2Q_simp
from pyzx.generate import cliffordT
import random
from copy import deepcopy
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimizer_logger import RandomCircuit_OptimizerLogger

if __name__ == '__main__':
    num_tests = 200
    qubits_gates = {}
    # qubits_gates[5] = [50, 100, 150, 200]
    # qubits_gates[10] = [100, 200, 300, 400]
    # qubits_gates[20] = [200, 400, 600, 800]
    # qubits_gates[40] = [400, 800, 1200, 1600]
    qubits_gates[40] = [1200, 1600]
    qubits_gates[80] = [800, 1600, 2400, 3200]
    save_path1 = "./rc_records/pyzx_full_reduce"
    save_path2 = "./rc_records/pyzx_cflow"
    os.makedirs(save_path1, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)
    for qubits, gate_list in qubits_gates.items():
        for gates in gate_list:
            log_dir1 = f"{save_path1}/{qubits}_{gates}"
            log_dir2 = f"{save_path2}/{qubits}_{gates}"
            os.makedirs(log_dir1, exist_ok=True)
            os.makedirs(log_dir2, exist_ok=True)
            logger1 = RandomCircuit_OptimizerLogger()
            logger2 = RandomCircuit_OptimizerLogger()
            logger1.set_initial("pyzx_full_reduce", qubits, gates)
            logger2.set_initial("pyzx_cflow", qubits, gates)
            for seed in range(num_tests):
                random.seed(seed)
                g = cliffordT(qubits, gates)
                original_circuit = zx.Circuit.from_graph(g).to_basic_gates()
                logger1.log_original_circuit_info(seed, len(original_circuit.gates), original_circuit.depth(), original_circuit.tcount(), original_circuit.twoqubitcount())
                logger2.log_original_circuit_info(seed, len(original_circuit.gates), original_circuit.depth(), original_circuit.tcount(), original_circuit.twoqubitcount())
                g_ = deepcopy(g)
                # full_reduce
                start = time()
                full_reduce(g_)
                optimized_c = zx.extract.extract_circuit(g_)
                optimized_c = zx.optimize.basic_optimization(optimized_c.to_basic_gates()).to_basic_gates()
                logger1.log_optimized_circuit_info(seed, len(optimized_c.gates), optimized_c.depth(), optimized_c.tcount(), optimized_c.twoqubitcount(), time() - start)
                
                g_ = deepcopy(g)
                # cflow
                start = time()
                zx.to_graph_like(g_)
                flow_2Q_simp(g_)
                optimized_c = zx.extract.extract_simple(g_)
                optimized_c = zx.optimize.basic_optimization(optimized_c.to_basic_gates()).to_basic_gates()
                logger2.log_optimized_circuit_info(seed, len(optimized_c.gates), optimized_c.depth(), optimized_c.tcount(), optimized_c.twoqubitcount(), time() - start)
            
            logger1.finish_log(log_dir1)
            logger2.finish_log(log_dir2)