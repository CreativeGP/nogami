from abc import ABC, abstractmethod
from typing import overload
import copy

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
from torch_geometric.nn import Sequential as geo_Sequential
import networkx as nx
from memory_profiler import profile

from src.util import Logger, forward_hook

# torch.distributions.categoricalを継承して、
class CategoricalMasked(torch.distributions.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, device="cpu"):
        self.device = device
        if masks is None:
            masks = []
        self.masks = masks
        if len(self.masks) != 0:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            # torch.where(condition, true, false)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(self.device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)

class AgentGNNBase(nn.Module):
    def get_value(self, graph, info):
        if self.args is not None and 'impl_light_feature' in self.args and  self.args.impl_light_feature:
            if isinstance(graph,(tuple,list)):
                value_obs = torch_geometric.data.Batch.from_data_list([self.get_value_feature_graph2(g) for g in graph])
                # for g in graph:
                #     _value_obs = self.get_value_feature_graph(g)
                #     _value_obs2 = self.get_value_feature_graph2(g)
                #     assert torch.all(_value_obs.x == _value_obs2.x)
                #     assert torch.all(_value_obs.edge_index == _value_obs2.edge_index)
                #     assert torch.all(_value_obs.edge_attr == _value_obs2.edge_attr)
            else:
                value_obs = self.get_value_feature_graph2(graph)
        else:
            if isinstance(graph,(tuple,list)):
                value_obs = torch_geometric.data.Batch.from_data_list([self.get_value_feature_graph(g) for g in graph])
            else:
                value_obs = self.get_value_feature_graph(graph)
            
        if 'agent' in self.args and self.args.agent == 'shared':
            if not isinstance(graph,(tuple,list,np.ndarray)):
                graph = [graph]
                info = [info]
            if self.args is not None and 'impl_light_feature' in self.args and self.args.impl_light_feature:
                value_obs = torch_geometric.data.Batch.from_data_list([self.get_policy_feature_graph2(g,i,mask_stop=False) for g, i in zip(graph,info)])
            else:
                value_obs = torch_geometric.data.Batch.from_data_list([self.get_policy_feature_graph(g,i,mask_stop=False) for g, i in zip(graph,info)])
        values = self.critic(value_obs)
        values = values.squeeze(-1)
        return values# 
    
    def load_checkpoint_at(*, step=0):
        
        self.load_state_dict(torch.load(f"checkpoint_{step}.pt"))

    def write_weight_logs(self, logger: Logger, global_step: int):
        pass
    
    def detailed_weight_logs(self, logger: Logger, global_step: int):
        for name, param in self.named_parameters():
            pascal_name = name.replace('.', '_')
            # バカみたいに多いのでwandbのみ出力
            logger.write_scalar(f'detailed_weights/{pascal_name}_mean', param.data.mean().item(), global_step, only_wandb=True)
            logger.write_scalar(f'detailed_weights/{pascal_name}_std', param.data.std().item(), global_step, only_wandb=True)
    
    def detailed_grad_logs(self, logger: Logger, global_step: int):
        for name, param in self.named_parameters():
            pascal_name = name.replace('.', '_')
            if param.grad is not None:
                logger.write_scalar(f'detailed_grads/{pascal_name}_mean', param.grad.abs().mean().item(), global_step, only_wandb=True)
                logger.write_scalar(f'detailed_grads/{pascal_name}_std', param.grad.abs().std().item(), global_step, only_wandb=True)

    # エージェント側で報酬を補正することもできる　
    def get_rewards(self, rewards: list, next_info: list[dict]) -> list:
        return rewards
        

# PPOで訓練できるAgentとして必要なもの
class PPOAgent(ABC):
    @abstractmethod
    def get_next_action(self, graph, info, action=None, device="cpu", testing=False, mask_stop=False):
        pass
    
    @abstractmethod
    def get_value(self, graph, info):
        pass

# PPGで訓練できるAgentとして必要なもの
class PPGAgent(ABC):
    @abstractmethod
    def get_next_action(self, graph, info, action=None, device="cpu", testing=False, mask_stop=False):
        pass
    
    @abstractmethod
    def get_value_from_critic_head(self, graph, info):
        pass
    
    @abstractmethod
    def get_value_from_actor_head(self, graph, info):
        pass
