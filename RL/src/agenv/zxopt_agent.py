from abc import ABC, abstractmethod
from typing import overload
import copy, argparse

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
from torch.distributions.categorical import Categorical
from torch_geometric.nn import Sequential as geo_Sequential
import networkx as nx
from memory_profiler import profile

from src.util import Logger, grad_statistics
from src.agenv.zxopt_agent1 import AgentGNN1
from src.agenv.zxopt_agent2 import AgentGNN2
from src.agenv.zxopt_agent3 import AgentGNN3



class Agent(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_features(self, obs):
        pass

    @abstractmethod
    def action_fn(feat):
        pass

    def get_next_action(self, obs, device):
        feat = self.calculate_features(obs)
        return action_fn(feat)

# class ActorCriticAgent(Agent):
#     def __init__(self, actor, critic):
#         self.actor = actor
#         self.critic = critic
    
#     def calculate_features(self, obs):
        

#     def get_action_and_value(self, obs, device):
#         feat = self.calculate_features(obs)
#         action, logprob = self.actor(feat)
#         value = self.critic(feat)
#         return action, logprob, None, value, None, None


def get_agent(envs, device, args, **kwargs):
    if type(args) == argparse.Namespace:
        agent = args.agent
    else:
        agent = args['agent']
    print("opening agent:", agent)
    if agent == "original":
        return AgentGNN1(envs, device, args, **kwargs)
    elif agent == "shared":
        return AgentGNN2(envs, device, args, **kwargs)
    elif agent == "ppg":
        return AgentGNN3(envs, device, args, **kwargs)

def get_agent_from_state_dict(envs, device, args: 'argparse.Namespace', state_dict: dict, **kwargs):
    agent = state_dict["agent"] if "agent" in state_dict else "original"
    args.agent = agent
    res = get_agent(envs, device, args, **kwargs)
    # load_state_dictは変なキーがあるとエラーを吐くので削除
    if "agent" in state_dict:
        del state_dict["agent"]
    res.load_state_dict(state_dict)
    return res

def get_agent_from_filename(envs, device, agent: str, model: str, **kwargs):
    res = get_agent(envs, device, {'agent': agent}, **kwargs)
    state_dict = torch.load(model, map_location=torch.device("cpu"))
        # load_state_dictは変なキーがあるとエラーを吐くので削除
    if "agent" in state_dict:
        del state_dict["agent"]
    res.load_state_dict(state_dict)
    return res

    
