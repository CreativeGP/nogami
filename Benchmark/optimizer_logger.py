from datetime import datetime
from time import time
import numpy as np
import pickle
import json
import os

class RandomCircuit_OptimizerLogger:
    def __init__(self):
        self.optimizer_ = None

    def set_initial(self, optimizer_name, qubits, gates):
        self.optimizer_ = optimizer_name
        self.time_stamp_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.qubits_ = qubits
        self.gates_ = gates
        self.original_circuit_records = {}
        self.optimized_circuit_records = {}

    def log_original_circuit_info(self, seed, gates, depth, tcount, twoqubits):
        record = {"seed" : seed
                , "gates" : gates
                , "depth" : depth
                , "tcount" : tcount
                , "twoqubits" : twoqubits
                , "cliffords" : gates - tcount
                , "non-cliffords" : tcount
                  }
        self.original_circuit_records[seed] = record

    def log_optimized_circuit_info(self, seed, gates, depth, tcount, twoqubits, execution_time):
        record = {"seed" : seed
                , "gates" : gates
                , "depth" : depth
                , "tcount" : tcount
                , "twoqubits" : twoqubits
                , "cliffords" : gates - tcount
                , "non-cliffords" : tcount
                , "execution time" : execution_time
                , "gates reduction rate" : (self.original_circuit_records[seed]["gates"] - gates)/self.original_circuit_records[seed]["gates"]
                , "depth reduction rate" : (self.original_circuit_records[seed]["depth"] - depth)/self.original_circuit_records[seed]["depth"]
                , "tcount reduction rate" : (self.original_circuit_records[seed]["tcount"] - tcount)/self.original_circuit_records[seed]["tcount"]
                , "twoqubits reduction rate" : (self.original_circuit_records[seed]["twoqubits"] - twoqubits)/self.original_circuit_records[seed]["twoqubits"]
                , "cliffords reduction rate" : (self.original_circuit_records[seed]["cliffords"] - (gates - tcount))/self.original_circuit_records[seed]["cliffords"]
                , "non-cliffords reduction rate" : (self.original_circuit_records[seed]["non-cliffords"] - tcount)/self.original_circuit_records[seed]["non-cliffords"]
                  }
        self.optimized_circuit_records[seed] = record

    def finish_log(self, save_path):
        g_list = []
        d_list = []
        t_list = []
        two_list = []
        c_list = []
        nc_list = []
        et_lit = []
        gr_list = []
        dr_list = []
        tr_list = []
        twor_list = []
        cr_list = []
        ncr_list = []
        for seed in self.original_circuit_records.keys():
            g_list.append(self.optimized_circuit_records[seed]["gates"])
            d_list.append(self.optimized_circuit_records[seed]["depth"])
            t_list.append(self.optimized_circuit_records[seed]["tcount"])
            two_list.append(self.optimized_circuit_records[seed]["twoqubits"])
            c_list.append(self.optimized_circuit_records[seed]["cliffords"])
            nc_list.append(self.optimized_circuit_records[seed]["non-cliffords"])
            et_lit.append(self.optimized_circuit_records[seed]["execution time"])
            gr_list.append(self.optimized_circuit_records[seed]["gates reduction rate"])
            dr_list.append(self.optimized_circuit_records[seed]["depth reduction rate"])
            tr_list.append(self.optimized_circuit_records[seed]["tcount reduction rate"])
            twor_list.append(self.optimized_circuit_records[seed]["twoqubits reduction rate"])
            cr_list.append(self.optimized_circuit_records[seed]["cliffords reduction rate"])
            ncr_list.append(self.optimized_circuit_records[seed]["non-cliffords reduction rate"])
        self.mean_records = {}
        self.mean_records["mean gates"] = np.mean(g_list)
        self.mean_records["mean depth"] = np.mean(d_list)
        self.mean_records["mean tcount"] = np.mean(t_list)
        self.mean_records["mean twoqubits"] = np.mean(two_list)
        self.mean_records["mean cliffords"] = np.mean(c_list)
        self.mean_records["mean non-cliffords"] = np.mean(nc_list)
        self.mean_records["mean execution time"] = np.mean(et_lit)
        self.mean_records["mean gates reduction rate"] = np.mean(gr_list)
        self.mean_records["mean depth reduction rate"] = np.mean(dr_list)
        self.mean_records["mean tcount reduction rate"] = np.mean(tr_list)
        self.mean_records["mean twoqubits reduction rate"] = np.mean(twor_list)
        self.mean_records["mean cliffords reduction rate"] = np.mean(cr_list)
        self.mean_records["mean non-cliffords reduction rate"] = np.mean(ncr_list)

        print("Overall information")
        print("time stamp: ", self.time_stamp_)
        print("qubits: ", self.qubits_)
        print("gates: ", self.gates_)
        print("mean gates: ", self.mean_records["mean gates"])
        print("mean depth: ", self.mean_records["mean depth"])
        print("mean tcount: ", self.mean_records["mean tcount"])
        print("mean twoqubits: ", self.mean_records["mean twoqubits"])
        print("mean cliffords: ", self.mean_records["mean cliffords"])
        print("mean non-cliffords: ", self.mean_records["mean non-cliffords"])
        print("mean execution time: ", self.mean_records["mean execution time"])
        print("mean gates reduction rate: ", self.mean_records["mean gates reduction rate"])
        print("mean depth reduction rate: ", self.mean_records["mean depth reduction rate"])
        print("mean tcount reduction rate: ", self.mean_records["mean tcount reduction rate"])
        print("mean twoqubits reduction rate: ", self.mean_records["mean twoqubits reduction rate"])
        print("mean cliffords reduction rate: ", self.mean_records["mean cliffords reduction rate"])
        print("mean non-cliffords reduction rate: ", self.mean_records["mean non-cliffords reduction rate"])
        filename = f"{self.optimizer_}_rc_{self.qubits_}_{self.gates_}.pkl"
        with open(os.path.join(save_path, filename), 'wb') as f:
            pickle.dump(self, f)
        
        filename = f"{self.optimizer_}_overallinfo_rc_{self.qubits_}_{self.gates_}.json"
        with open(os.path.join(save_path, filename), 'w', encoding="utf-8") as f:
            json.dump(self.mean_records, f, ensure_ascii=False, indent=4)
        print("Log saved as ", filename)

    @classmethod
    def load_log(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def get_original_circuit_records(self):
        return self.original_circuit_records
    
    def get_optimized_circuit_records(self):
        return self.optimized_circuit_records
    
    def get_mean_records(self):
        return self.mean_records