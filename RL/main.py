import os
import argparse
import time
from distutils.util import strtobool
from socket import gethostname

import torch
import gym

from src.util import rootdir

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-type", type=str, default="gates",
        help="the type of gate to optimize for")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="zx-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=8983440,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    
    # NOTE(cgp): wandbを使っているコードは少なくとも公開されていない
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--use-async", type=bool, default=False, help="use parallel?")
    parser.add_argument("--num-process", type=int, default=8,
        help="the number of multiprocesses") #default 8
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments") #default 8
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.05,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args




count = 0
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"



def make_env(gym_id, seed, idx, capture_video, run_name, qubits, depth, gate_type):
    def thunk():
        env = gym.make(gym_id, qubits=qubits, depth=depth, gate_type=gate_type)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, rootdir(f"/videos/{run_name}"))
        return env

    return thunk
gym.envs.registration.register(
    id='zx-v0',
    # entry_point=lambda qubit, depth: ZXEnv(qubit, depth),
    entry_point=f"src.agenv.zxopt_env:ZXEnv",
)
print("__name__", __name__)

if __name__ == "__main__":
    from src.training_method.ppo import PPO
    from src.agenv.zxopt_agent import AgentGNN
    from src.util import CustomizedAsyncVectorEnv, CustomizedSyncVectorEnv

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #Training size
    qubits = 5
    depth = 55
    run_name = f"{args.gym_id}__{args.exp_name}__{gethostname()}__{args.seed}__{int(time.time())}"
    if args.use_async:
        envs = CustomizedAsyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, qubits, depth, args.gate_type) for i in range(args.num_envs)],
            shared_memory=False
        )
    else:
        envs = CustomizedSyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, qubits, depth, args.gate_type) for i in range(args.num_envs)],
        )

    agent = AgentGNN(envs, device).to(device)

    ppo = PPO(envs, agent, args, run_name)

    ppo.run()
       




