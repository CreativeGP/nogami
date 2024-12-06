import os
import sys
import argparse
import json
import random
from distutils.util import strtobool

import numpy as np
import pyzx as zx
import torch
from torch_geometric.data import Batch, Data

# NOTE(cgp): あまりよくないらしいけど、ルートモジュールより上を経由するにはこうするしかないかも
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.util import rootdir, CustomizedSyncVectorEnv
from src.agenv.zxopt_agent import get_agent_from_state_dict

global device

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grading-data", type=str, default="/home/wsl/Research/nogami/zx/RL/test/grading_data/state_dict_3784704_zx-v0__main__8983440__1727192464_model5x70_gates_new/")
    parser.add_argument("--gate-type", type=str, default="gates")
    parser.add_argument("-n", type=int, default=-1,)
    parser.add_argument("--use-space", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,)
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,)
    return parser.parse_known_args()[0]

import json
import multiprocessing as mp
import argparse
args = parse_args()

if __name__ == "__main__":
    # arguments
    if args.grading_data.endswith("/"):
        args.grading_data = args.grading_data[:-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    print("Grade for model: ", args.grading_data)
    print("Gate type: ", args.gate_type)
    print()
    if args.verbose:
        if args.use_space:
            print("qubit depth rl(mean) rl(std) bo(mean) bo(std) pyzx(mean) pyzx(std) win(mean)\n")
        else:
            print("qubit\tdepth\trl(mean)\trl(std)\tbo(mean)\tbo(std)\tpyzx(mean)\tpyzx(std)\twin(mean)\n")
    else:
        if args.use_space:
            print("qubit depth gr(mean) gr(std) win(mean)\n")
        else:
            print("qubit\tgate\tgr(mean)\tgr(std)\twin(mean)\t\n")

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
                with open(f"{args.grading_data}/bo_stats_stopping_{qubit}x{depth}.json") as f:
                    bo_stats = json.load(f)            
                with open(f"{args.grading_data}/initial_stats_stopping_{qubit}x{depth}.json") as f:
                    initial_stats = json.load(f)
                with open(f"{args.grading_data}/pyzx_stats_stopping_{qubit}x{depth}.json") as f:
                    pyzx_stats = json.load(f)
                rl_reduction = np.array(rl_stats[args.gate_type]) - np.array(initial_stats[args.gate_type])
                pyzx_reduction = np.array(pyzx_stats[args.gate_type]) - np.array(initial_stats[args.gate_type])
                bo_reduction = np.array(bo_stats[args.gate_type]) - np.array(initial_stats[args.gate_type])
                wins = []
                for rl, pyzx in zip(rl_stats[args.gate_type], pyzx_stats[args.gate_type]):
                    wins.append(1 if rl < pyzx else -1)
                
                n = 0
                if args.n == -1:
                    n = len(rl_reduction)
                else:
                    n = args.n
                    rl_reduction = random.sample(list(rl_reduction), args.n)
                    pyzx_reduction = random.sample(list(pyzx_reduction), args.n)
                    bo_reduction = random.sample(list(bo_reduction), args.n)
                    wins = random.sample(list(wins), args.n)

                wins_percentage = np.array(wins)*50 + 50

                if args.verbose:
                    if args.use_space:
                        print(f"{qubit} {depth} {np.mean(rl_reduction):.3f} {np.std(rl_reduction):.3f} {np.mean(bo_reduction):.3f} {np.std(bo_reduction):.3f} {np.mean(pyzx_reduction):.3f} {np.std(pyzx_reduction):.3f} {np.mean(wins_percentage):.0f} {n}")
                    else:
                        print(f"{qubit}\t{depth}\t{np.mean(rl_reduction):.3f}\t{np.std(rl_reduction):.3f}\t{np.mean(bo_reduction):.3f}\t{np.std(bo_reduction):.3f}\t{np.mean(pyzx_reduction):.3f}\t{np.std(pyzx_reduction):.3f}\t{np.mean(wins):.3f}\t{n}")
                else:
                    if args.use_space:
                        print(f"{qubit} {depth} {np.mean(rl_reduction):.3f} {np.std(rl_reduction):.3f} {np.mean(wins_percentage):.0f} {n}")
                    else:
                        print(f"{qubit}\t{depth}\t{np.mean(rl_reduction):.3f}\t{np.std(rl_reduction):.3f}\t{np.mean(wins):.3f}\t{n}")
            except Exception as e:
                print(e)
