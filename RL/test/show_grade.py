import os
import sys
import argparse
import json
from distutils.util import strtobool

import pandas as pd
import numpy as np
import pyzx as zx
import torch
from torch_geometric.data import Batch, Data

# NOTE(cgp): あまりよくないらしいけど、ルートモジュールより上を経由するにはこうするしかないかも
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.util import rootdir, CustomizedSyncVectorEnv
from src.agenv.zxopt_agent import AgentGNN

global device
device = torch.device("cuda")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grading-data", type=str,)
    parser.add_argument("--gate-type", type=str,)
    parser.add_argument("-n", type=int, default=-1,)
    parser.add_argument("--use-space", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)

    return parser.parse_known_args()[0]

import json
import multiprocessing as mp
import argparse
args = parse_args()

if __name__ == "__main__":
    # arguments
    if args.grading_data.endswith("/"):
        args.grading_data = args.grading_data[:-1]

    print("Grade for model: ", args.grading_data)
    print("Gate type: ", args.gate_type)
    print()
    if args.use_space:
        print("qubit depth mean std n")
    else:
        print("qubit gate\tmean\tstd\tn")

    qubits_depths = {
        5: [55, 105, 155, 210],
        10: [90, 180, 270, 360],
        20: [160, 325, 485, 650],
        40: [290, 580, 875, 1165],
        80: [525,1050,1575,2100]
    }
    for qubit, depths in qubits_depths.items():
        for depth in depths:
            try:
                with open(f"{args.grading_data}/rl_stats_stopping_{qubit}x{depth}.json") as f:
                    rl_stats = json.load(f)            
                with open(f"{args.grading_data}/initial_stats_stopping_{qubit}x{depth}.json") as f:
                    initial_stats = json.load(f)
                reduction = np.array(rl_stats[args.gate_type]) - np.array(initial_stats[args.gate_type])
                if args.n == -1:
                    args.n = len(reduction)
                reduction = np.random.choice(reduction, args.n)

                if args.use_space:
                    print(f"{qubit} {depth} {np.mean(reduction):.3f} {np.std(reduction):.3f} {args.n}")
                else:
                    print(f"{qubit} {depth}\t{np.mean(reduction):.3f}\t{np.std(reduction):.3f}\t{args.n}")
            except Exception as e:
                pass
