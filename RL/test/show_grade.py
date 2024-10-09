import os
import sys
import argparse
import json
import random
import time
from distutils.util import strtobool

import gym
import gym_zx
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
                print(f"{qubit} {depth}\t{np.mean(reduction):.3f}\t{np.std(reduction):.3f}\t{args.n}")
            except Exception as e:
                pass
