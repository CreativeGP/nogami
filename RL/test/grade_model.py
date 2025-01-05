import os
import sys
import argparse
import json
import random
import time
from distutils.util import strtobool

import gym
import pandas as pd
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

    parser.add_argument("--model", type=str, default="/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_104448_zx-v0__main__8983440__1726934959_model5x70_gates_new.pt")
    parser.add_argument("--gate-type", type=str, default="gates")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments") 
    parser.add_argument("--num-episodes", type=int, default=1000,
        help="the number of episodes to run")
    parser.add_argument("--gym-id", type=str, default="zx-v0",
        help="the id of the gym environment")
    
    parser.add_argument("--qubit", type=int, default=-1, help="指定がなければすべてのqubitについて評価します.")
    parser.add_argument("--depth", type=int, default=-1,)

    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--seed", type=int, default=10000,
        help="seed of the experiment")
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


def get_results(param):
    global args
    rl_time, full_reduce_time = [], []
    qubits, depth = param

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    run_name = "HP5"
    capture_video = False
    envs = CustomizedSyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, capture_video, run_name, qubits, depth, args.gate_type) for i in range(args.num_envs)]
    )

    agent = get_agent_from_state_dict(envs, device, args, torch.load(args.model, map_location=torch.device("cpu"))).to(device)  

    agent.eval()
   
    done = False
    wins = 0
    rl_stats = {
        "gates": [],
        "tcount": [],
        "clifford": [],
        "CNOT": [],
        "CX": [],
        "CZ": [],
        "had": [],
        "twoqubits": [],
        "min_gates": [],
        "opt_episode_len": [],
        "episode_len": [],
        "opt_episode_len": [],
        "initial_2q": [],
        "action_stats": [],
        "depth": [],
        "initial_depth": [],
    }
    pyzx_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubits": []}
    bo_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubits": []}
    initial_stats = {
        "gates": [],
        "tcount": [],
        "clifford": [],
        "CNOT": [],
        "CX": [],
        "CZ": [],
        "had": [],
        "twoqubits": [],
    }
    rl_action_pattern = pd.DataFrame()
    for episode in range(args.num_episodes):  
        print(episode)
        done = False
        obs0, reset_info = envs.reset()
        full_reduce_time.append(reset_info[0]["full_reduce_time"])
        state, info = [reset_info[0]["graph_obs"]], reset_info

        start = time.time()
        while not done:
            action, _, _, _, action_id, _ = agent.get_next_action(state, info, device=device)
            action = action.flatten()
            state, reward, done, deprecated, info = envs.step(action_id.cpu().numpy())
        end = time.time()

        rl_time.append(end - start)
        info = info[0]["final_info"]
        rl_circ_s = info["rl_stats"]
        bo_circ_s = info["bo_stats"]
        no_opt_s = info["no_opt_stats"]
        zx_circ_s = info["pyzx_stats"]
        in_circ_s = info["initial_stats"]

        rl_stats["gates"].append(rl_circ_s["gates"])

        rl_stats["tcount"].append(rl_circ_s["tcount"])
        rl_stats["clifford"].append(rl_circ_s["clifford"])
        rl_stats["CNOT"].append(rl_circ_s["CNOT"])
        rl_stats["CZ"].append(rl_circ_s["CZ"])
        rl_stats["had"].append(rl_circ_s["had"])
        rl_stats["twoqubits"].append(rl_circ_s["twoqubits"])
        rl_stats["initial_2q"].append(no_opt_s["twoqubits"])
        rl_stats["episode_len"].append(info["episode_len"])
        rl_stats["opt_episode_len"].append(info["opt_episode_len"] + info["episode_len"])
        rl_stats["action_stats"].append(info["action_stats"])
        rl_stats["initial_depth"].append(info["initial_depth"])
        rl_stats["depth"].append(info["depth"])

        # print(info["action_pattern"])
        
        action_pattern_df = pd.DataFrame(info["action_pattern"])
        
        action_pattern_df["Episode"] = [episode]*action_pattern_df.shape[0]
        rl_action_pattern = pd.concat([rl_action_pattern, action_pattern_df], ignore_index=True)

        bo_stats["gates"].append(bo_circ_s["gates"])

        bo_stats["tcount"].append(bo_circ_s["tcount"])
        bo_stats["clifford"].append(bo_circ_s["clifford"])
        bo_stats["CNOT"].append(bo_circ_s["CNOT"])
        bo_stats["CZ"].append(bo_circ_s["CZ"])
        bo_stats["had"].append(bo_circ_s["had"])
        bo_stats["twoqubits"].append(bo_circ_s["twoqubits"])

        pyzx_stats["gates"].append(zx_circ_s["gates"])
        pyzx_stats["tcount"].append(zx_circ_s["tcount"])
        pyzx_stats["clifford"].append(zx_circ_s["clifford"])
        pyzx_stats["CNOT"].append(zx_circ_s["CNOT"])
        pyzx_stats["CZ"].append(zx_circ_s["CZ"])
        pyzx_stats["had"].append(zx_circ_s["had"])
        pyzx_stats["twoqubits"].append(zx_circ_s["twoqubits"])

        initial_stats["gates"].append(in_circ_s["gates"])
        initial_stats["tcount"].append(in_circ_s["tcount"])
        initial_stats["clifford"].append(in_circ_s["clifford"])
        initial_stats["CNOT"].append(in_circ_s["CNOT"])
        initial_stats["CZ"].append(in_circ_s["CZ"])
        initial_stats["had"].append(in_circ_s["had"])
        initial_stats["twoqubits"].append(in_circ_s["twoqubits"])

        wins += info["win_vs_pyzx"]

        print("Gates with RL", sum(rl_stats["gates"]) / len(rl_stats["gates"]))
        print("Gates with PyZX", sum(pyzx_stats["gates"]) / len(pyzx_stats["gates"]))
        print("Gates with BOpt", sum(bo_stats["gates"]) / len(bo_stats["gates"]))
        print("2q with RL", sum(rl_stats["twoqubits"]) / len(rl_stats["twoqubits"]))
        print("2q with PyZX", sum(pyzx_stats["twoqubits"]) / len(pyzx_stats["twoqubits"]))
        print("2q with BOpt", sum(bo_stats["twoqubits"]) / len(bo_stats["twoqubits"]))
        print("2q initial", sum(rl_stats["initial_2q"]) / len(rl_stats["initial_2q"]))
        print("Wins:", wins)

    if not os.path.exists(rootdir(f"/test/grading_data/{model_file_name}")):
        os.makedirs(rootdir(f"/test/grading_data/{model_file_name}"))
    rl_action_pattern.to_csv(rootdir(f"/test/grading_data/{model_file_name}/action_pattern_{qubits}x{depth}.json"), index=False)  
   
    with open(rootdir(f"/test/grading_data/{model_file_name}/rl_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(rl_stats, f)
    with open(rootdir(f"/test/grading_data/{model_file_name}/pyzx_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(pyzx_stats, f)

    with open(rootdir(f"/test/grading_data/{model_file_name}/initial_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(initial_stats, f)
    with open(rootdir(f"/test/grading_data/{model_file_name}/bo_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(bo_stats, f)
    
    return np.mean(full_reduce_time), np.mean(rl_time), np.std(full_reduce_time), np.std(rl_time)


if __name__ == "__main__":
    # arguments
    gym.envs.registration.register(
        id='zx-v0',
        # entry_point=lambda qubit, depth: ZXEnv(qubit, depth),
        entry_point=f"src.agenv.zxopt_env:ZXEnv",
    )

    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else torch.device("cpu")

    qubits = [5]
    depths = [55,110]
    qubits_depths = {
        5: [55, 105, 155, 210],
        10: [90, 180, 270, 360],
        20: [160, 325, 485, 650],
        40: [290, 580, 875, 1165],
        80: [525,1050,1575,2100]
    }

    if args.qubit != -1 and args.depth != -1:
        qubits_depths = {args.qubit: [args.depth]}

    for qubit, depths in qubits_depths.items():
        fr_time_depth, rl_time_depth = [],[]
        fr_time_var, rl_time_var = [],[]
        for depth in depths:
            print(f"Qubits, Depth {qubit, depth}")
            fr_time, rl_time, fr_var, rl_var = get_results((qubit,depth))
            fr_time_depth.append(fr_time)
            rl_time_depth.append(rl_time)
            fr_time_var.append(fr_var)
            rl_time_var.append(rl_var)
            print(rl_time, rl_var)
            
        with open(rootdir(f"/test/grading_data/{model_file_name}/time_depth_{qubit}x{depth}.json"), 'w') as f:
            json.dump({"full_time":fr_time_depth, "rl_time":rl_time_depth, "full_var":fr_time_var, "rl_var":rl_time_var}, f)
