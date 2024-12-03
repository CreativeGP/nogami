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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.util import rootdir, CustomizedSyncVectorEnv
from src.agenv.zxopt_agent import get_agent_from_state_dict

global device

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model", type=str, default="/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_104448_zx-v0__main__8983440__1726934959_model5x70_gates_new.pt")
    parser.add_argument("--model", type=str, default="/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_5369856_zx-v0__training2-with-vtargetvar-add__8983440__1730560905.pt")
    parser.add_argument("--cliffordT", type=int, nargs=2, default=[5,55], help="two integers for Clifford+T optimization")
    # parser.add_argument("--qasm", type=str, default="")
    parser.add_argument("--qasm", type=str, default=None,)#default="/home/wsl/Research/nogami/zx/qasm/mod_adder_1024.qasm")
    parser.add_argument("-n", type=int, default=1, help="number of circuits")
    parser.add_argument("--stop-at", type=int, default=-1, help="RLエージェントが終了する操作数を強制的に設定する")

    parser.add_argument("--gate-type", type=str, default="gates")
    # parser.add_argument("--num-envs", type=int, default=1,
    #     help="the number of parallel game environments") 
    # parser.add_argument("--num-episodes", type=int, default=1000,
    #     help="the number of episodes to run")
    parser.add_argument("--gym-id", type=str, default="zx-v0",
        help="the id of the gym environment")

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

def make_env(gym_id, seed, idx, capture_video, run_name, gate_type, g):
    def thunk():
        env = gym.make(gym_id, gate_type = gate_type, g = g)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, rootdir(f"/videos/{run_name}"))
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def get_results(agent,i):
    global args
    rl_time, full_reduce_time = [], []

    if args.seed != 10000:
        random.seed(args.seed+i)
        np.random.seed(args.seed+i)
        torch.manual_seed(args.seed+i)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    else:
        random.seed(random.randint(0, 1000000))
        np.random.seed(random.randint(0, 1000000))
        torch.manual_seed(random.randint(0, 1000000))
        torch.backends.cudnn.deterministic = args.torch_deterministic


    run_name = "HP5"
    capture_video = False
    if args.qasm is not None:
        circuit = zx.Circuit.load(args.qasm).to_basic_gates()
        g = circuit.to_graph()
    elif args.cliffordT is not None:
        print(args.cliffordT)
        g = zx.generate.cliffordT(args.cliffordT[0], args.cliffordT[1])
        circuit = zx.Circuit.from_graph(g).to_basic_gates()
    qubits, depth = circuit.qubits, len(circuit.gates)
    print("Qubits:", qubits, "Depth:", depth)
    print("Depth:", circuit.depth(), "Tcount:", circuit.tcount())

    envs = CustomizedSyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, capture_video, run_name, args.gate_type, g) for i in range(1)]
    )
    agent.envs = envs

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
    done = False
    obs0, reset_info = envs.reset()
    full_reduce_time.append(reset_info[0]["full_reduce_time"])
    state, info = [reset_info[0]["graph_obs"]], reset_info

    start = time.time()
    count = 0
    while not done:
        action, _, _, _, action_id = agent.get_next_action(state, info, device=device, mask_stop=args.stop_at > 0)
        action = action.flatten()
        if args.stop_at > 0 and count >= args.stop_at:
            # 強制終了
            state, reward, done, deprecated, info = envs.step(np.array([0]))
        else:
            state, reward, done, deprecated, info = envs.step(action_id.cpu().numpy())
        count += 1
    end = time.time()

    wins = info[0]["final_info"]["win_vs_pyzx"]
    final_circuit = info[0]["final_info"]["final_circuit"]

    rl_time.append(end - start)
    info = info[0]["final_info"]
    rl_circ_s = info["rl_stats"]
    bo_circ_s = info["bo_stats"]
    no_opt_s = info["no_opt_stats"]
    zx_circ_s = info["pyzx_stats"]
    in_circ_s = info["initial_stats"]

    print("Action pattern: ", info["action_pattern"])
    print("Gates NoOpt", no_opt_s["gates"])
    print("Gates initial", in_circ_s["gates"])
    print("Gates with BOpt", bo_circ_s["gates"])
    print("Gates with RL", rl_circ_s["gates"])
    print("Gates with PyZX", zx_circ_s["gates"])
    print("2q with RL", rl_circ_s["twoqubits"])
    print("2q with PyZX", zx_circ_s["twoqubits"])
    print("2q with BOpt", bo_circ_s["twoqubits"])
    print("2q initial", in_circ_s["twoqubits"])
    print("Wins:", wins)

    print("Depth:", final_circuit.depth(), "Tcount:", final_circuit.tcount())


    return np.mean(full_reduce_time), np.mean(rl_time), np.std(full_reduce_time), np.std(rl_time)

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # arguments
    gym.envs.registration.register(
        id='zx-v0',
        # entry_point=lambda qubit, depth: ZXEnv(qubit, depth),
        entry_point=f"src.agenv.zxopt_env:ZXEnvForTest",
    )
    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else torch.device("cpu")

    agent = get_agent_from_state_dict(None, device, args, torch.load(args.model, map_location=torch.device("cpu"))).to(device)
    agent.eval()

    for i in range(args.n):
        get_results(agent,i)
