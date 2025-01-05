import time

import torch
from torch_geometric.data import Batch, Data
import gym

from src.util import Logger, Timer, rootdir, count_autograd_graph, for_minibatches, print_grad_summary, print_random_states, register_all_forward_hooks


class Trajectory():
    def __init__(self, envs, agent, logger=None, device="cuda"):
        assert issubclass(type(envs), gym.vector.VectorEnv), "envs must be a gym.vector.VectorEnv"
        self.envs = envs
        self.agent = agent
        self.collect_value = False
        self.device = device
        self.length = 0
        self.logger = logger
        if hasattr(self.agent, "get_value"):
            self.collect_value = True
    
    def reset(self):
        self.length = 0
        self.global_step = 0
        start_time = time.time()
        self.s0, self.next_info = self.envs.reset()
        self.next_s = self.s0
        self.next_done = torch.zeros(self.envs.num_envs).to(self.device)

    
    # self.states, actions, logprobs, rewards, dones, valuesまでのデータを収集
    def collect(self, length):
        self.length = length
        self.actions = torch.zeros((length, self.envs.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((length, self.envs.num_envs)).to(self.device)
        self.rewards = torch.zeros((length, self.envs.num_envs)).to(self.device)
        self.dones = torch.zeros((length, self.envs.num_envs)).to(self.device)
        self.values = torch.zeros((length, self.envs.num_envs)).to(self.device)    

        self.states = []
        self.infos = []
        print("Step loop started:", length, "steps in total.")
        for step in range(length):  
            # self.timer.start()
            # print(random.random())

            self.global_step += 1 * self.envs.num_envs
            self.states.extend(self.next_s)
            self.infos.extend(self.next_info)
            self.dones[step] = self.next_done

            with torch.no_grad():
                action, logprob, _, logits, action_ids, _ = self.agent.get_next_action(self.next_s, self.next_info, device=self.device)
                if self.collect_value:
                    self.values[step] = self.agent.get_value(self.next_s, self.next_info).flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            # print()
            # print("before env step")
            # print_random_states(show_hash=True)

            # 観測情報を取得 (s a r s')
            s_, r, done, deprecated, self.next_info = self.envs.step(action_ids.cpu().numpy())

            self.rewards[step] = torch.tensor(r).to(self.device).view(-1)
            self.next_done = torch.Tensor(done).to(self.device)
            self.next_s = s_

            # logging
            if not self.logger is None:
                for idx, info in enumerate(self.next_info):
                    if 'action' in info and not info['action'] is None:
                        self.logger.data['action_counter'].append(info['action'])
                    if 'nodes' in info and not info['nodes'] is None:
                        self.logger.data['action_nodes'].extend(info['nodes'])

                    if 'final_info' in info:
                        item = info['final_info']
                        if done[idx]:
                            # print(f"self.global_step={self.global_step}, episodic_return={item['episode']['r']}")
                            self.logger.data['cumulative_reward'].append(item["episode"]["r"])
                            self.logger.data['cumulative_episode_length'].append(item["episode"]["l"])
                            self.logger.data['remaining_pivot_size'].append(item["remaining_pivot_size"])
                            self.logger.data['remaining_lcomp_size'].append(item["remaining_lcomp_size"])
                            self.logger.data['cumulative_max_reward_difference'].append(item["max_reward_difference"])
                            self.logger.data['action_patterns'].append(item["action_pattern"])
                            self.logger.data['action_counter'].append(item["action"])
                            self.logger.data['optimal_episode_length'].append(item["opt_episode_len"])
                            self.logger.data['pyzx_gates'].append(item["pyzx_gates"])
                            self.logger.data['rl_gates'].append(item["rl_gates"])
                            self.logger.data['swap_gates'].append(item["swap_cost"])
                            self.logger.data['pyzx_swap_gates'].append(item["pyzx_swap_cost"])
                            self.logger.data['wins_vs_pyzx'].append(item["win_vs_pyzx"])
        self.b_logprobs = self.logprobs.reshape(-1)
        self.b_rewards = self.rewards.reshape(-1)
        self.b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        self.b_values = self.values.reshape(-1)
    
    # advantageとreturnを計算
    def calculate_advantages(self, gamma, use_gae=True, gae_lambda=0):
        # self.timer.start()

        # bootstrap value if not done, implement GAE-Lambda advantage calculation
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_s, self.next_info).reshape(1, -1)
            if use_gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.length)):
                    if t == self.length - 1:
                        nextnonterminal = 1.0 - self.next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.length-1)):
                    if t == self.length - 1:
                        nextnonterminal = 1.0 - self.next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - self.values
        # self.timer.stop()
        # print(self.timer)
        # self.timer.clear()
        self.advantages = advantages
        self.returns = returns
        self.b_advantages = advantages.reshape(-1)
        self.b_returns = returns.reshape(-1)

        # print(self.rewards.mean())
