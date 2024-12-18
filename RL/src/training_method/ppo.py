import random, time, sys, signal
from socket import gethostname
import shutil

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

from distutils.util import strtobool
from torch_geometric.data import Batch, Data


from src.util import Logger, Timer, rootdir, count_autograd_graph, for_minibatches
from src.training_method.trajectory import Trajectory

class PPO():
    def __init__(self, envs, agent, args, run_name):
        assert issubclass(type(envs), gym.vector.VectorEnv), "envs must be a gym.vector.VectorEnv"
        assert hasattr(agent, "get_next_action"), "agent must have a `get_next_action` method"
        assert hasattr(agent, "get_value"), "agent must have a `get_value` method"
        # self.actor = ActorNetwork()
        # self.critic = CriticNetwork()
        # self.env = Env()
        self.gamma = 0.99
        self.epsilon = 0.2
        self.lr = 1e-4
        self.batch_size = 64
        self.epochs = 10
        self.clip = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


        self.args = args
        self.run_name = run_name
        self.logger = Logger(run_name, args, use_wandb=True if 'use-wandb' in args and args.wandb else False)

        # print(random.random())
        self.envs = envs
        self.agent = agent
        # print(self.agent.state_dict())
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        
        self.traj = Trajectory(self.envs, self.agent, logger=self.logger, device=self.device)
        self.traj.reset()
        if args.checkpoint is not None:
            self.traj.global_step = args.global_step
        num_updates = self.args.total_timesteps // self.args.batch_size
        self.start_update = self.traj.global_step // self.args.batch_size if args.checkpoint is not None else 0

        self.init_logger_data()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    
    def __del__(self):
        self.clean_up_log()
    
    def signal_handler(self, sig, frame):
        self.clean_up_log()
        sys.exit(0)

    def clean_up_log(self):
        print(self.run_name)
        if self.args.checkpoint is not None:
            if self.traj.global_step-self.args.global_step < 8000:
                # remove directory
                print(f"Global steps {self.traj.global_step} is less than 8000. Removing directory...")
                shutil.rmtree(rootdir(f"/runs/{self.run_name}"))
        else:
            if self.traj.global_step < 8000:
                # remove directory
                print(f"Global steps {self.traj.global_step} is less than 8000. Removing directory...")
                shutil.rmtree(rootdir(f"/runs/{self.run_name}"))

    
    def init_logger_data(self):
        self.logger.data['cumulative_reward'] = []
        self.logger.data['cumulative_episode_length'] = []
        self.logger.data['action_counter'] = []
        self.logger.data['action_nodes'] = []
        self.logger.data['remaining_pivot_size'] = []
        self.logger.data['remaining_lcomp_size'] = []
        self.logger.data['cumulative_max_reward_difference'] = []
        self.logger.data['action_patterns'] = []
        self.logger.data['optimal_episode_length'] = []
        self.logger.data['pyzx_gates'] = []
        self.logger.data['rl_gates'] = []
        self.logger.data['swap_gates'] = []
        self.logger.data['pyzx_swap_gates'] = []
        self.logger.data['wins_vs_pyzx'] = []

    def run(self):
        self.timer = Timer()
        self.start_time = time.time()

        print(f"Run start: {self.run_name}")


        NUM_UPDATES = 2048
        for update in range(self.start_update, NUM_UPDATES):
            if update % 10 == 1:
                state_dict = self.agent.state_dict()
                state_dict['agent'] = self.args.agent if self.args.agent is not None else "original"
                torch.save(state_dict, rootdir(f"/checkpoints/state_dict_{self.run_name}_{self.traj.global_step}_model5x70_gates_new.pt"))
            if self.args.anneal_lr:
                frac = max(1.0 / 100, 1.0 - (update - 1.0) / (NUM_UPDATES * 5.0 / 6))
                lrnow = frac * self.args.learning_rate
                print(lrnow)
                self.optimizer.param_groups[0]["lr"] = lrnow
                neg_reward_discount = max(1, 5 * (1 - 4 * update / NUM_UPDATES))
            if update * 1.0 / NUM_UPDATES > 5.0 / 6:
                self.ent_coef = 0
            else:
                self.ent_coef = self.args.ent_coef
            
            self.traj.collect(self.args.num_steps)
            self.traj.calculate_advantages(self.args.gamma, self.args.gae, self.args.gae_lambda)
            # old_params = [param.clone() for param in self.agent.parameters()]

            self.update_networks()

            # new_params = [param.clone() for param in self.agent.parameters()]
            # diff_params = []
            # for o, n in zip(old_params, new_params):
            #     diff_params.append((o - n))
            # print("[", end="")
            # for i, d in enumerate(diff_params):
            #     print(f"[{d.mean()}, {d.std()}], ", end="")
            # print("]")

            # exit(0)


            self.write_other_logs()
            print(
                'rl_gates: ',
                sum(self.logger.data['rl_gates']) / len(self.logger.data['rl_gates']),
                ' pyzx_gates: ',
                sum(self.logger.data['pyzx_gates']) / len(self.logger.data['pyzx_gates']),
                ' wins: ',
                sum(self.logger.data['wins_vs_pyzx']) / len(self.logger.data['wins_vs_pyzx']),
            )


            self.init_logger_data()
            
            # states = self.collect_data()
            # self.calculate_advantages()
            # self.update_networks()
        self.envs.close()
        self.logger.close()

        state_dict = self.agent.state_dict()
        state_dict['agent'] = self.args.agent if self.args.agent is not None else "original"
        torch.save(state_dict, "state_dict_model5x70_twoqubits_new.pt")

    def update_networks(self):
        print(np.count_nonzero(self.traj.rewards.cpu().numpy() < -0.1))

        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)  
        clipfracs = []
        print("Epoch loop started:", self.args.update_epochs, "epochs in total.")
        for epoch in range(self.args.update_epochs):
            self.timer.start()
            np.random.shuffle(b_inds)  

            for start in range(
                0, self.args.batch_size, self.args.minibatch_size
            ):  # loop over entire batch, one minibatch at the time
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]
                states_batch = [self.traj.states[i] for i in mb_inds]
                infos_batch = [self.traj.infos[i] for i in mb_inds]
                # バッチでとってきたものの一つ先を見る, actionを指定しておくことでlogprobを取る
                _, newlogprob, entropy, logits, _ = self.agent.get_next_action(
                    states_batch,
                    infos_batch,
                    action=self.traj.b_actions.long()[mb_inds].T, device=self.device
                )
                logratio = newlogprob - self.traj.b_logprobs[mb_inds]  # logratio = log(newprob/oldprob)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = self.traj.b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = self.agent.get_value(states_batch, infos_batch)
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - self.traj.b_returns[mb_inds]) ** 2
                    v_clipped = self.traj.b_values[mb_inds] + torch.clamp(
                        newvalue - self.traj.b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.traj.b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - self.traj.b_returns[mb_inds]) ** 2).mean()
                
                # spike detection
                if v_loss > 0.01:
                    pass

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                # show_autograd_graph(loss.grad_fn)
                # print(count_autograd_graph(loss.grad_fn))
                # return
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()


            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    break
            self.timer.stop()
        print(self.timer)
        self.timer.clear()


        # logging
        y_pred, y_true = self.traj.b_values.cpu().numpy(), self.traj.b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        vtarget_variance =  np.var(y_true - y_pred)
        var_val = np.var(y_pred)
        var_ret = np.var(y_true)
        cov_pred_true = np.cov(y_pred, y_true)[0,1]
        self.logger.write_scalar("losses/vtarget_variance", vtarget_variance, self.traj.global_step)
        self.logger.write_scalar("losses/var_values", var_val, self.traj.global_step)
        self.logger.write_scalar("losses/var_returns", var_ret, self.traj.global_step)
        self.logger.write_scalar("losses/cov_val_ret", cov_pred_true, self.traj.global_step)

        
        self.logger.write_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.traj.global_step)
        self.logger.write_scalar("losses/value_loss", v_loss.item(), self.traj.global_step)
        self.logger.write_scalar("losses/policy_loss", pg_loss.item(), self.traj.global_step)
        self.logger.write_scalar("losses/entropy", entropy_loss.item(), self.traj.global_step)
        self.logger.write_scalar("losses/old_approx_kl", old_approx_kl.item(), self.traj.global_step)
        self.logger.write_scalar("losses/approx_kl", approx_kl.item(), self.traj.global_step)
        self.logger.write_scalar("losses/clipfrac", np.mean(clipfracs), self.traj.global_step)
        self.logger.write_scalar("losses/explained_variance", explained_var, self.traj.global_step)
        self.logger.write_scalar("charts/value_function", torch.mean(self.traj.b_values), self.traj.global_step)
        self.logger.writer.add_histogram(
            "histograms/value_function",
            self.traj.b_values.cpu()
            .detach()
            .numpy()
            .reshape(
                -1,
            ),
            self.traj.global_step,
        )
        self.logger.writer.add_histogram(
            "histograms/logits",
            logits.cpu()
            .detach()
            .numpy()
            .reshape(
                -1,
            ),
            self.traj.global_step,
        )
    
    def write_other_logs(self):
        self.logger.write_scalar('charts/SPS', int(self.traj.global_step / (time.time() - self.start_time)), self.traj.global_step)

        self.logger.write_mean('charts/mean_reward', 'cumulative_reward', self.traj.global_step)
        self.logger.write_mean('charts/mean_episode_length', 'cumulative_episode_length', self.traj.global_step)
        self.logger.write_mean('charts/max_reward_difference_mean', 'cumulative_max_reward_difference', self.traj.global_step)
        self.logger.write_mean('charts/remaining_pivot_size_mean', 'remaining_pivot_size', self.traj.global_step)
        self.logger.write_mean('charts/remaining_lcomp_size_mean', 'remaining_lcomp_size', self.traj.global_step)
        self.logger.write_mean('charts/opt_episode_len_mean', 'optimal_episode_length', self.traj.global_step)
        self.logger.write_mean('charts/pyzx_gates', 'pyzx_gates', self.traj.global_step)
        self.logger.write_mean('charts/rl_gates', 'rl_gates', self.traj.global_step)
        self.logger.write_mean('charts/wins_vs_pyzx', 'wins_vs_pyzx', self.traj.global_step)
        self.logger.write_mean('charts/swap_gates', 'swap_gates', self.traj.global_step)
        self.logger.write_mean('charts/pyzx_swap_gates', 'pyzx_swap_gates', self.traj.global_step)

        self.agent.write_weight_logs(self.logger, self.traj.global_step)

        self.logger.write_histogram('histograms/reward_distribution', 'cumulative_reward', self.traj.global_step)
        self.logger.writer.add_histogram('histograms/step_reward_distribution', np.array(self.traj.b_rewards.cpu()), self.traj.global_step)
        self.logger.writer.add_histogram('histograms/step_return_distribution', np.array(self.traj.b_returns.cpu()), self.traj.global_step)
        self.logger.writer.add_histogram('histograms/step_value_distribution', np.array(self.traj.b_values.cpu()), self.traj.global_step)
        self.logger.writer.add_histogram('histograms/estimate_advantage_distribution', np.array(self.traj.b_advantages.cpu()), self.traj.global_step)
        self.logger.write_histogram('histograms/episode_length_distribution', 'cumulative_episode_length', self.traj.global_step)
        self.logger.write_histogram('histograms/action_counter_distribution', 'action_counter', self.traj.global_step)
        self.logger.write_histogram('histograms/action_nodes_distribution', 'action_nodes', self.traj.global_step)
        self.logger.write_histogram('histograms/remaining_pivot_size_distribution', 'remaining_pivot_size', self.traj.global_step)
        self.logger.write_histogram('histograms/remaining_lcomp_size_distribution', 'remaining_lcomp_size', self.traj.global_step)
        self.logger.write_histogram('histograms/max_reward_difference_distribution', 'cumulative_max_reward_difference', self.traj.global_step)
        self.logger.write_histogram('histograms/opt_episode_len_distribution', 'optimal_episode_length', self.traj.global_step)
        self.logger.write_histogram('histograms/rl_gates', 'rl_gates', self.traj.global_step)
        self.logger.write_histogram('histograms/pyzx_gates', 'pyzx_gates', self.traj.global_step)

