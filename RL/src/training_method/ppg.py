import random, time, sys, signal

import numpy as np
import torch
import torch.nn as nn

from src.agenv.zxopt_agent import AgentGNN3
from src.training_method.ppo import PPO
from src.util import Logger, Timer, rootdir, count_autograd_graph, for_minibatches

class PPG(PPO):
    class ReuseSample():
        def __init__(self):
            self.value = 0
            self.qvalue = 0
            self.state = 0
            self.info = 0

    def __init__(self, envs, agent: AgentGNN3, args, run_name):
        self.config: dict = {
            "ppo_epochs": 8,
            "n_policy_phase": 1,
            "aux_epochs": 4,
            "kl": 'whole',
            "lr_aux_policy": args.learning_rate,
            "lr_aux_value": args.learning_rate,
            "β_clone": 0.1,
        }
        self.lr_aux_policy: float = args.learning_rate
        self.lr_aux_value: float = args.learning_rate
        self.B: List[PPG.ReuseSample] = []
        super().__init__(envs, agent, args, run_name)
        self.logger.write_dict("ppo", self.config)
        print("PPG Config:", self.config)


    def run(self):
        self.timer = Timer()
        self.start_time = time.time()

        print(f"Run start: {self.run_name}")


        NUM_UPDATES = self.args.total_timesteps // self.args.batch_size
        for update in range(self.start_update, NUM_UPDATES):
            if update % 10 == 1:
                state_dict = self.agent.state_dict()
                state_dict['agent'] = self.args.agent if self.args.agent is not None else "original"
                torch.save(state_dict, rootdir(f"/checkpoints/state_dict_{self.run_name}_{self.traj.global_step}_model5x70_gates_new.pt"))
            if self.args.anneal_lr:
                frac = max(1.0 / 100, 1.0 - (update - 1.0) / (NUM_UPDATES * 5.0 / 6))
                self.optimizer.param_groups[0]["lr"] = frac * self.args.learning_rate
                self.lr_aux_policy = frac * self.config['lr_aux_policy']
                self.lr_aux_value = frac * self.config['lr_aux_value']
                neg_reward_discount = max(1, 5 * (1 - 4 * update / NUM_UPDATES))
            if update * 1.0 / NUM_UPDATES > 5.0 / 6:
                self.ent_coef = 0
            else:
                self.ent_coef = self.args.ent_coef
            
            self.traj.collect(self.args.num_steps)
            self.traj.calculate_advantages(self.args.gamma, self.args.gae, self.args.gae_lambda)
            # for n
            #   update policy network for E_pi
            #   update value network for E_v
            #   add all (s_t, V_t) to B
            # Compute and store pi_old(|s_t) for all s_t in B
            # for E_aux
            #   update L_joint
            #   update L_value
            self.update_networks()

            for i in range(self.args.batch_size):
                sample = PPG.ReuseSample()
                sample.value = self.traj.b_values[i]
                sample.qvalue = self.traj.b_returns[i]
                sample.state = self.traj.states[i]
                sample.info = self.traj.infos[i]
                self.B += [sample]

            if (update+1) % self.config["n_policy_phase"] == 0:
                self.aux_network_update()
                self.B = []
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
        self.envs.close()
        self.logger.close()

        state_dict = self.agent.state_dict()
        state_dict['agent'] = self.args.agent if self.args.agent is not None else "original"
        torch.save(state_dict, "state_dict_model5x70_twoqubits_new.pt")

    def update_networks(self):
        print(np.count_nonzero(self.traj.rewards.cpu().numpy() < -0.1))

        # Optimizing the policy and value network
        clipfracs = []
        print("Epoch loop started:", self.args.update_epochs, "epochs in total.")
        for epoch in range(self.args.update_epochs):
            self.timer.start()

            for mb_inds in for_minibatches(self.args.batch_size, self.args.minibatch_size):  # loop over entire batch, one minibatch at the time
                states_batch = [self.traj.states[i] for i in mb_inds]
                infos_batch = [self.traj.infos[i] for i in mb_inds]
                # バッチでとってきたものの一つ先を見る
                _, newlogprob, entropy, logits, _, _ = self.agent.get_next_action(
                    states_batch,
                    infos_batch,
                    self.traj.b_actions.long()[mb_inds].T, device=self.device
                )
                newvalue = self.agent.get_value_from_critic_head(states_batch, infos_batch)
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
    
    def calculate_kl(self, a_logprobs, b_logprobs):
        logratio = a_logprobs - b_logprobs
        ratio = logratio.exp()
        if self.config["schulman_kl"]:
            return ((ratio - 1) - logratio).mean()
        else:
            return (-logratio).mean()

    
    def aux_network_update(self):
        B_states = np.array([sample.state for sample in self.B])
        B_infos = np.array([sample.info for sample in self.B])
        B_values = np.array([sample.value for sample in self.B])
        B_qs = np.array([sample.qvalue for sample in self.B])

        B_logprob = np.array([])
        B_pd = np.array([])
        for inds in np.array_split(np.arange(len(self.B)), len(self.B) // self.args.minibatch_size + 1):
            # 現在のπで計算
            _,  _B_logprob, _, _, _, pd = self.agent.get_next_action(
                B_states[inds],
                B_infos[inds],
                action=None,
                device=self.device
            )
            pd = [torch.distributions.Categorical(logits=logit) for logit in pd.logits]
            B_logprob = np.concatenate([B_logprob, _B_logprob.detach().numpy()])
            B_pd = np.concatenate([B_pd, pd])
        print(B_logprob.shape, B_logprob)
        B_logprob = torch.Tensor(B_logprob)

        dummy_step = 0
        for _ in range(self.config["aux_epochs"]):
            for mb_inds in for_minibatches(len(self.B), self.args.minibatch_size):
                states_batch =  np.array(B_states[mb_inds])
                infos_batch =  np.array(B_infos[mb_inds])
                values_batch =  torch.Tensor(B_values[mb_inds])
                qs_batch =  torch.Tensor(B_qs[mb_inds])

                _, newlogprob, _, _, _, pd = self.agent.get_next_action(
                    states_batch,
                    infos_batch,
                    action=None,
                    device=self.device
                )
                pd = [torch.distributions.Categorical(logits=logit) for logit in pd.logits]


                newvalue = self.agent.get_value_from_actor_head(states_batch, infos_batch)
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - values_batch) ** 2
                    v_clipped = values_batch + torch.clamp(
                        newvalue - values_batch,
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - qs_batch) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    L_aux = 0.5 * v_loss_max.mean()
                else:
                    L_aux = 0.5 * ((newvalue - qs_batch) ** 2).mean()

                # print("newlogprob", newlogprob)
                # print("B_logprob", B_logprob[mb_inds])
                logratio = newlogprob - B_logprob[mb_inds]
                ratio = logratio.exp()
                if self.config["kl"] == 'naive':
                    new_kl = ((ratio - 1) - logratio).mean() # スパイクたちがち
                    L_kl = new_kl
                elif self.config["kl"] == 'schulman':
                    old_kl = (-logratio).mean()
                    L_kl = old_kl
                elif self.config["kl"] == 'whole':
                    whole_kl = torch.Tensor([torch.distributions.kl.kl_divergence(pd1, pd2) for pd1, pd2 in zip(B_pd[mb_inds], pd)]).mean()
                    # 一個一個やるとちょっと遅いかもしれん
                    L_kl = whole_kl

                L_joint = L_aux + self.config["β_clone"]*L_kl
                print("L_joint = L_aux + β*L_kl", L_joint.item(), L_aux.item(), self.config["β_clone"]*L_kl.item())
                # import sys; sys.exit()

                self.optimizer.param_groups[0]["lr"] = self.lr_aux_policy
                self.optimizer.zero_grad()
                L_joint.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                # Value loss
                newvalue = self.agent.get_value_from_critic_head(states_batch, infos_batch)
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - values_batch) ** 2
                    v_clipped = values_batch + torch.clamp(
                        newvalue - values_batch,
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - qs_batch) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    L_value = 0.5 * v_loss_max.mean()
                else:
                    L_value = 0.5 * ((newvalue - qs_batch) ** 2).mean()
                print("L_value", L_value)

                self.optimizer.param_groups[0]["lr"] = self.lr_aux_value
                self.optimizer.zero_grad()
                L_value.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                # 特別ログ
                self.logger.write_scalar("losses/ppg_laux", L_aux.item(), self.traj.global_step+dummy_step)
                self.logger.write_scalar("losses/ppg_lkl", L_kl.item(), self.traj.global_step+dummy_step)
                self.logger.write_scalar("losses/ppg_lvalue", L_value.item(), self.traj.global_step+dummy_step)
                dummy_step += 1




        # logging Laux, Lkl?


