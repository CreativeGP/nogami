import os
import sys
import argparse
import json
import random
import time
from distutils.util import strtobool
import json
import multiprocessing as mp
import argparse

import gym
import pandas as pd
import numpy as np
import pyzx as zx
import torch
import dill
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
    parser.add_argument("--num-process", type=int, default=1,
        help="the number of parallel game environments") 
    parser.add_argument("--num-episodes", type=int, default=1000,
        help="the number of episodes to run")
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

args = parse_args()
model_file_name = args.model.split("/")[-1].split('.')[0]
filedir = os.path.dirname(__file__)

class DillProcess(mp.Process):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)  # Save the target function as bytes, using dill

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)    # Unpickle the target function before executing
            self._target(*self._args, **self._kwargs)  # Execute the target function





class MpData():
    def __init__(self):
        self.wins = 0
        self.rl_time = []
        self.rl_stats = {
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
        self.pyzx_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubits": []}
        self.bo_stats = {"gates": [], "tcount": [], "clifford": [], "CNOT": [], "CX": [], "CZ": [], "had": [], "twoqubits": []}
        self.initial_stats = {
            "gates": [],
            "tcount": [],
            "clifford": [],
            "CNOT": [],
            "CX": [],
            "CZ": [],
            "had": [],
            "twoqubits": [],
        }
        self.rl_action_pattern = pd.DataFrame()
        self.full_reduce_time = []

    def concat(self, local):
        self.wins += local.wins
        self.rl_time.extend(local.rl_time)
        self.full_reduce_time.extend(local.full_reduce_time)
        for key in self.rl_stats:
            self.rl_stats[key].extend(local.rl_stats[key])
        for key in self.pyzx_stats:
            self.pyzx_stats[key].extend(local.pyzx_stats[key])
        for key in self.rlbo_stats_stats:
            self.bo_stats[key].extend(local.bo_stats[key])
        for key in self.initial_stats:
            self.initial_stats[key].extend(local.initial_stats[key])
        self.rl_action_pattern = pd.concat([self.rl_action_pattern, local.rl_action_pattern], ignore_index=True)


# NOTE(cgp): envs, agentも共有したいけど、dillでpickleできないので、ここで定義
def mp_worker(loop_num, args, master):
    def make_env(gym_id, seed, idx, capture_video, run_name, qubits, gates, gate_type):
        env = gym.make(gym_id, qubits = qubits, depth = gates, gate_type = gate_type)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, rootdir(f"/videos/{run_name}"))
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env


    run_name = "HP5"
    capture_video = False
    # envs = CustomizedSyncVectorEnv(
    #     [(lambda: make_env(args.gym_id, args.seed + i, i, capture_video, run_name, qubits, depth, args.gate_type)) for i in range(args.num_envs)]
    # )
    qubits, depth = 5, 60
    envs = make_env(args.gym_id, args.seed, 0, capture_video, run_name, qubits, depth, args.gate_type)

    agent = get_agent_from_state_dict(envs, device, args, torch.load(args.model, map_location=torch.device("cpu"))).to(device)  
    agent.eval()


    local = MpData()
    for episode in range(loop_num):  
        done = False
        obs0, reset_info = envs.reset()
        local.full_reduce_time.append(reset_info[0]["full_reduce_time"])
        state, info = [reset_info[0]["graph_obs"]], reset_info

        start = time.time()
        while not done:
            action, _, _, _, action_id = agent.get_next_action(state, info, device=device)
            action = action.flatten()
            state, reward, done, deprecated, info = envs.step(action_id.cpu().numpy())
        end = time.time()

        local.rl_time.append(end - start)
        info = info["final_info"]
        rl_circ_s = info["rl_stats"]
        bo_circ_s = info["bo_stats"]
        no_opt_s = info["no_opt_stats"]
        zx_circ_s = info["pyzx_stats"]
        in_circ_s = info["initial_stats"]

        local.rl_stats["gates"].append(rl_circ_s["gates"])

        local.rl_stats["tcount"].append(rl_circ_s["tcount"])
        local.rl_stats["clifford"].append(rl_circ_s["clifford"])
        local.rl_stats["CNOT"].append(rl_circ_s["CNOT"])
        local.rl_stats["CZ"].append(rl_circ_s["CZ"])
        local.rl_stats["had"].append(rl_circ_s["had"])
        local.rl_stats["twoqubits"].append(rl_circ_s["twoqubits"])
        local.rl_stats["initial_2q"].append(no_opt_s["twoqubits"])
        local.rl_stats["episode_len"].append(info["episode_len"])
        local.rl_stats["opt_episode_len"].append(info["opt_episode_len"] + info["episode_len"])
        local.rl_stats["action_stats"].append(info["action_stats"])
        local.rl_stats["initial_depth"].append(info["initial_depth"])
        local.rl_stats["depth"].append(info["depth"])

        # print(info["action_pattern"])
        
        action_pattern_df = pd.DataFrame(info["action_pattern"])
        
        action_pattern_df["Episode"] = [episode]*action_pattern_df.shape[0]
        local.rl_action_pattern = pd.concat([local.rl_action_pattern, action_pattern_df], ignore_index=True)

        local.bo_stats["gates"].append(bo_circ_s["gates"])

        local.bo_stats["tcount"].append(bo_circ_s["tcount"])
        local.bo_stats["clifford"].append(bo_circ_s["clifford"])
        local.bo_stats["CNOT"].append(bo_circ_s["CNOT"])
        local.bo_stats["CZ"].append(bo_circ_s["CZ"])
        local.bo_stats["had"].append(bo_circ_s["had"])
        local.bo_stats["twoqubits"].append(bo_circ_s["twoqubits"])

        local.pyzx_stats["gates"].append(zx_circ_s["gates"])
        local.pyzx_stats["tcount"].append(zx_circ_s["tcount"])
        local.pyzx_stats["clifford"].append(zx_circ_s["clifford"])
        local.pyzx_stats["CNOT"].append(zx_circ_s["CNOT"])
        local.pyzx_stats["CZ"].append(zx_circ_s["CZ"])
        local.pyzx_stats["had"].append(zx_circ_s["had"])
        local.pyzx_stats["twoqubits"].append(zx_circ_s["twoqubits"])

        local.initial_stats["gates"].append(in_circ_s["gates"])
        local.initial_stats["tcount"].append(in_circ_s["tcount"])
        local.initial_stats["clifford"].append(in_circ_s["clifford"])
        local.initial_stats["CNOT"].append(in_circ_s["CNOT"])
        local.initial_stats["CZ"].append(in_circ_s["CZ"])
        local.initial_stats["had"].append(in_circ_s["had"])
        local.initial_stats["twoqubits"].append(in_circ_s["twoqubits"])

        local.wins += info["win_vs_pyzx"]

        # print("Gates with RL", sum(rl_stats["gates"]) / len(rl_stats["gates"]))
        # print("Gates with PyZX", sum(pyzx_stats["gates"]) / len(pyzx_stats["gates"]))
        # print("Gates with BOpt", sum(bo_stats["gates"]) / len(bo_stats["gates"]))
        # print("2q with RL", sum(rl_stats["twoqubits"]) / len(rl_stats["twoqubits"]))
        # print("2q with PyZX", sum(pyzx_stats["twoqubits"]) / len(pyzx_stats["twoqubits"]))
        # print("2q with BOpt", sum(bo_stats["twoqubits"]) / len(bo_stats["twoqubits"]))
        # print("2q initial", sum(rl_stats["initial_2q"]) / len(rl_stats["initial_2q"]))
        # print("Wins:", wins)
    master.concat(local)


def get_results(param):
    global args
    qubits, depth = param

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # run_name = "HP5"
    # capture_video = False
    # # envs = CustomizedSyncVectorEnv(
    # #     [(lambda: make_env(args.gym_id, args.seed + i, i, capture_video, run_name, qubits, depth, args.gate_type)) for i in range(args.num_envs)]
    # # )
    # envs = make_env(args.gym_id, args.seed, 0, capture_video, run_name, qubits, depth, args.gate_type)

    # agent = AgentGNN(envs, device).to(device)  

    # agent.load_state_dict(
    #     torch.load(args.model, map_location=torch.device("cpu"))
    # )
    # agent.eval()
   
    master = MpData()
    process = []
    loop_num = args.num_episodes//args.num_process
    mp.set_start_method("spawn")
    for i in range(args.num_process):
        # NOTE(cgp): 平均取ることを想定してるから、seedは厳密にやらなくていいよね
        p = DillProcess(target=mp_worker, args=(loop_num, args, master))
        p.start()
        process.append(p)
        # seed += 25*i
    for p in process:
        p.join()


    if not os.path.exists(rootdir(f"/grading_data/{model_file_name}")):
        os.makedirs(rootdir(f"/grading_data/{model_file_name}"))
    master.rl_action_pattern.to_csv(rootdir(f"/grading_data/{model_file_name}/action_pattern_{qubits}x{depth}.json"), index=False)  
   
    with open(rootdir(f"/grading_data/{model_file_name}/rl_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(master.rl_stats, f)
    with open(rootdir(f"/grading_data/{model_file_name}/pyzx_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(master.pyzx_stats, f)

    with open(rootdir(f"/grading_data/{model_file_name}/initial_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(master.initial_stats, f)
    with open(rootdir(f"/grading_data/{model_file_name}/bo_stats_stopping_{qubits}x{depth}.json"), "w") as f:
        json.dump(master.bo_stats, f)
    
    return np.mean(master.full_reduce_time), np.mean(master.rl_time), np.std(master.full_reduce_time), np.std(master.rl_time)


if __name__ == "__main__":
    # arguments
    gym.envs.registration.register(
        id='zx-v0',
        # entry_point=lambda qubit, depth: ZXEnv(qubit, depth),
        entry_point=f"src.agenv.zxopt_env:ZXEnv",
    )
    device = torch.device("cuda") if torch.cuda.is_available() and args.cuda else torch.device("cpu")

    print(model_file_name)

    qubits = [5]
    depths = [55,110]
    qubits_depths = {
        5: [55, 105, 155, 210],
        10: [90, 180, 270, 360],
        20: [160, 325, 485, 650],
        40: [290, 580, 875, 1165],
        80: [525,1050,1575,2100]
    }
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
            
        with open(rootdir(f"/grading_data/{model_file_name}/time_depth_{qubit}x{depth}.json")) as f:
            json.dump({"full_time":fr_time_depth, "rl_time":rl_time_depth, "full_var":fr_time_var, "rl_var":rl_time_var}, f)
