import os, sys, argparse, json, random, time, copy
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
from src.agenv.zxopt_env import ActionHistory

global device

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model", type=str, default="/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_104448_zx-v0__main__8983440__1726934959_model5x70_gates_new.pt")
    parser.add_argument("--model", type=str, default="/home/wsl/Research/nogami/zx/RL/checkpoints/state_dict_5369856_zx-v0__training2-with-vtargetvar-add__8983440__1730560905.pt")
    parser.add_argument("--cliffordT", type=int, nargs=2, default=[5,50], help="two integers for Clifford+T optimization")
    # parser.add_argument("--qasm", type=str, default="")
    parser.add_argument("--qasm", type=str, default=None,)#default="/home/wsl/Research/nogami/zx/qasm/mod_adder_1024.qasm")
    parser.add_argument("--max-traversal", type=int, default=100000, help="max number of traversal")

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
    parser.add_argument("--seed", type=int, default=42,#default=10000,
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
rndstate_for_circuit_generation = None

def make_env(gym_id, seed, idx, capture_video, run_name, gate_type, g):
    def thunk():
        env = gym.make(gym_id, gate_type = gate_type, g = g, silent=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, rootdir(f"/videos/{run_name}"))
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def get_results(agent,i):
    global args, rndstate_for_circuit_generation
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
    if rndstate_for_circuit_generation is None:
        rndstate_for_circuit_generation = random.getstate()

    run_name = "HP5"
    capture_video = False
    if args.qasm is not None:
        circuit = zx.Circuit.load(args.qasm).to_basic_gates()
        g = circuit.to_graph()
    elif args.cliffordT is not None:
        print(args.cliffordT)
        random.setstate(rndstate_for_circuit_generation)
        g = zx.generate.cliffordT(args.cliffordT[0], args.cliffordT[1])
        circuit = zx.Circuit.from_graph(g).to_basic_gates()
        rndstate_for_circuit_generation = random.getstate()
    
    qubits, depth = circuit.qubits, len(circuit.gates)
    print("Qubits:", qubits, "Gates", depth)
    print("Depth:", circuit.depth(), "Tcount:", circuit.tcount())


    class TraversalNode:
        def __init__(self):
            self.graph: zx.Graph
            self.info: dict
            self.depth: int = 0
            self.gate_reduction: int = 0
            self.parent: "TraversalNode" = None
            self.children: list["TraversalNode"] = []
            self.history: list[ActionHistory] = []
        
    def print_history(node: TraversalNode):
        arr = []
        while node is not None:
            arr.append(node)
            node = node.parent
        for n in reversed(arr):
            if len(n.history) == 0:
                continue
            e = n.history[-1]
            print(e.act, e.vs[0] if len(e.vs) > 0 else "", e.vs[1] if len(e.vs) > 1 else "")
    def load_history(s: str) -> list[ActionHistory]:
        return json.loads(s)

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
    # envs = CustomizedSyncVectorEnv(
    #     [make_env(args.gym_id, args.seed + i, i, capture_video, run_name, args.gate_type, g) for i in range(1)]
    # )
    # agent.envs = envs
    # obs0, reset_info = envs.reset()
    # # full_reduce_time.append(reset_info[0]["full_reduce_time"])
    # state, info = [reset_info[0]["graph_obs"]], reset_info
    # root_node = TraversalNode()
    # root_node.graph = state[0]
    # root_node.info = info[0]
    # root_node.depth = 0
    # root_gates = reset_info[0]['circuit_data'][args.gate_type]
    # traversal_list = [root_node]

    env = gym.make('zx-v0', gate_type=args.gate_type, g=g, silent=True)
    # full_reduce_time.append(reset_info[0]["full_reduce_time"])
    obs0, reset_info = env.reset()
    root_node = TraversalNode()
    root_node.graph = reset_info['graph_obs']
    root_node.info = reset_info
    root_node.depth = 0
    root_gates = reset_info['circuit_data'][args.gate_type]
    traversal_list = [root_node]
    good_traversal_list = []

    print("ROOT", root_node.graph.graph)

    LONG = 100

    max_gr = 0
    def sample_next():
        nonlocal max_gr
        nonlocal traversal_list, good_traversal_list, LONG

        if len(traversal_list) > LONG:
            LONG += 5
            traversal_list = traversal_list[0:int(LONG/5)]+traversal_list[int(LONG/2):int(LONG)]

        dist = [(node.gate_reduction+1 if node.gate_reduction > 0 else 1)**3 * 0.001 for node in traversal_list]
        grs = [node.gate_reduction for node in traversal_list]
        # print(grs)
        if max(grs) > max_gr:
            node = traversal_list[grs.index(max(grs))]
            print()
            print_history(node)
            while node is not None:
                print(f"{node.gate_reduction}.", end="")
                node = node.parent
            print()
            max_gr = max(grs)
        return random.choices(traversal_list, weights=dist, k=1)[0]
        
    
    state, info = [reset_info['graph_obs']], [reset_info]

    start = time.time()
    count = 0
    while len(traversal_list) > 0 and count < args.max_traversal:
        node = sample_next()
        b_state, b_info = [node.graph], [node.info]
        action, _, _, _, action_id, _ = agent.get_next_action(b_state, b_info, device=device)
        action = action.flatten()

        new_node = TraversalNode()
        new_node.graph = copy.deepcopy(b_state)[0]
        assert isinstance(env.env.env, gym.Env), "gymは変なラッパーをかませてるから、確実にgym.Envにアクセスして"
        env.env.env.graph = new_node.graph
        env.env.env.current_gates = root_gates - node.gate_reduction
        env.env.env.set_info(b_info[0])
        # NOTE: こいつはなかでenv.graphを書き換えるので注意
        state, reward, done, deprecated, info = env.step(action_id.cpu().numpy())
        new_node.graph = state
        new_node.info = info
        new_node.depth = node.depth + 1
        new_node.parent = node

        if done:
            traversal_list.remove(node)
        else:
            new_node.gate_reduction = root_gates - info['circuit_data'][args.gate_type]
            new_node.history = node.history.copy() + [copy.copy(info['history'])]
            if new_node.parent is not None:
                print(new_node.gate_reduction-new_node.parent.gate_reduction, end=" ", flush=True)
            if True:
                node.children.append(new_node)
                traversal_list.append(new_node)

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

    get_results(agent, 0)
