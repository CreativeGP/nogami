import os, sys, argparse, json, random, time
from distutils.util import strtobool

import gym
import pandas as pd
import numpy as np
import pyzx as zx
import torch
from torch_geometric.data import Batch, Data

# NOTE(cgp): あまりよくないらしいけど、ルートモジュールより上を経由するにはこうするしかないかも
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.util import rootdir, CustomizedSyncVectorEnv, print_weights
from src.agenv.zxopt_agent import get_agent_from_state_dict, get_agent
from src.agenv.zxopt_env import ZXEnvForTest

global device

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_104448_zx-v0__main__8983440__1726934959_model5x70_gates_new.pt")
    parser.add_argument("--agent", type=str, default="original")
    # parser.add_argument("--num-steps", type=int, default=128,
    #     help="the number of steps to run in each environment per policy rollout")

    return parser.parse_known_args()[0]

import json
import multiprocessing as mp
import argparse
args = parse_args()
model_file_name = args.model.split("/")[-1].split('.')[0]
filedir = os.path.dirname(__file__)

def make_env(gym_id, seed, idx, capture_video, run_name, qubits, gates, gate_type):
    def thunk():
        env = gym.make(gym_id, qubits = qubits, depth = gates, gate_type = gate_type)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, rootdir(f"/videos/{run_name}"))
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

device = 'cpu'

if __name__ == "__main__":
    agent = get_agent_from_state_dict(None, device, args, torch.load(args.model, map_location=torch.device("cpu"))).to(device)
    print_weights(agent, 'critic_ff.4.weight')

    for _ in range(30):
        g_circ = zx.generate.cliffordT(5, 55)
        env = ZXEnvForTest(g_circ, gate_type="gates", silent=False, args=args)
        _, info = env.reset()
        print(agent.critic(agent.get_value_feature_graph(g_circ,)).item())
