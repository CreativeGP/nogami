from time import time
import pyzx as zx
from pyzx.generate import cliffordT
import random
from copy import deepcopy
import os
import sys
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.transpiler import CouplingMap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimizer_logger import RandomCircuit_OptimizerLogger

if __name__ == '__main__':
    num_tests = 200
    qubits_gates = {}
    qubits_gates[5] = [50, 100, 150, 200]
    qubits_gates[10] = [100, 200, 300, 400]
    qubits_gates[20] = [200, 400, 600, 800]
    qubits_gates[40] = [400, 800, 1200, 1600]
    qubits_gates[80] = [800, 1600, 2400, 3200]
    save_path1 = "./rc_records/qiskit"
    os.makedirs(save_path1, exist_ok=True)
    for qubits, gate_list in qubits_gates.items():
        all_to_all_coupling = CouplingMap.from_full(qubits)
        for gates in gate_list:
            log_dir1 = f"{save_path1}/{qubits}_{gates}"
            os.makedirs(log_dir1, exist_ok=True)
            logger1 = RandomCircuit_OptimizerLogger()
            logger1.set_initial("qiskit", qubits, gates)
            for seed in range(num_tests):
                random.seed(seed)
                g = cliffordT(qubits, gates)
                original_circuit = zx.Circuit.from_graph(g).to_basic_gates()
                logger1.log_original_circuit_info(seed, len(original_circuit.gates), original_circuit.depth(), original_circuit.tcount(), original_circuit.twoqubitcount())
                qc = QuantumCircuit.from_qasm_str(original_circuit.to_qasm())
                
                # qiskit optimization
                # qubit topology = all-to-all
                start = time()
                optimized_circuit = transpile(qc, coupling_map=all_to_all_coupling, optimization_level=3)
                duration = time() - start
                
                optimized_c = zx.Circuit.from_qasm(qasm2.dumps(optimized_circuit)).to_basic_gates()
                logger1.log_optimized_circuit_info(seed, len(optimized_c.gates), optimized_c.depth(), optimized_c.tcount(), optimized_c.twoqubitcount(), duration)
            logger1.finish_log(log_dir1)