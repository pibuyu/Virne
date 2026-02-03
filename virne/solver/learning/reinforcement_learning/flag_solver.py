from collections import Counter
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch.distributions import Categorical
from torch_geometric.data import Batch
from omegaconf import open_dict

from virne.solver import SolverRegistry
from virne.network import AttributeBenchmarkManager
from virne.solver.learning.rl_core import NodePairStepInstanceRLEnv, PPOSolver, InstanceAgent
from virne.solver.learning.rl_core.buffer import RolloutBuffer
from virne.solver.learning.utils import apply_mask_to_logit, get_pyg_data
from virne.solver.learning.rl_policy.flag_vne_policy import FlagVneActorCritic
from virne.solver.learning.neural_network import GCNConvNet, GATConvNet


class FlagVneInstanceEnv(NodePairStepInstanceRLEnv):
    """
    Environment used by FlagVNE. It mirrors the implementation in flag-vne.
    """
    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        with open_dict(config):
            config.solver.node_ranking_method = 'nrm'
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)

        benchmarks = AttributeBenchmarkManager.get_from_cache('p_net')
        if benchmarks is None:
            benchmarks = AttributeBenchmarkManager.get_benchmarks(p_net, node_attrs=True, link_attrs=True, link_sum_attrs=True)
        self.node_attr_benchmarks = benchmarks.node_attr_benchmarks
        self.link_attr_benchmarks = benchmarks.link_attr_benchmarks
        self.link_sum_attr_benchmarks = benchmarks.link_sum_attr_benchmarks
        self._calculate_graph_metrics()

    def _calculate_graph_metrics(self):
        p_net_node_degrees = np.array([list(nx.degree_centrality(self.p_net).values())], dtype=np.float32).T
        denom = p_net_node_degrees.max() - p_net_node_degrees.min()
        if denom == 0:
            self.p_net_node_degrees = p_net_node_degrees
        else:
            self.p_net_node_degrees = (p_net_node_degrees - p_net_node_degrees.min()) / denom

    @property
    def next_unplaced_v_node_id(self):
        if self.num_placed_v_net_nodes == self.v_net.num_nodes:
            return 0
        return int(self.v_net.ranked_nodes[self.num_placed_v_net_nodes])

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        action_mask = self.generate_action_mask().flatten()
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
            'curr_v_node_id': self.next_unplaced_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'action_mask': action_mask,
        }

    def _get_p_net_obs(self):
        attr_type_list = ['resource']
        node_data = self.obs_handler.get_node_attrs_obs(
            self.p_net,
            node_attr_types=attr_type_list,
            node_attr_benchmarks=self.node_attr_benchmarks
        )
        p_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(
            self.p_net,
            link_attr_types=attr_type_list,
            aggr='max',
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(
            self.p_net,
            link_attr_types=attr_type_list,
            aggr='sum',
            link_sum_attr_benchmarks=self.link_sum_attr_benchmarks
        )
        p_node_link_mean_resource = self.obs_handler.get_link_aggr_attrs_obs(
            self.p_net,
            link_attr_types=attr_type_list,
            aggr='mean',
            link_sum_attr_benchmarks=self.link_attr_benchmarks
        )
        p_nodes_status = self.obs_handler.get_p_net_nodes_status(
            self.p_net,
            self.v_net,
            self.solution['node_slots']
        )
        v_node_sizes = np.ones((self.p_net.num_nodes, 1), dtype=np.float32) * self.v_net.num_nodes / 10
        v_num_placed_nodes = np.ones((self.p_net.num_nodes, 1), dtype=np.float32) * self.num_placed_v_net_nodes / 10
        avg_distance = self.obs_handler.get_average_distance(self.p_net, self.solution['node_slots'], normalization=True)
        node_data = np.concatenate(
            (
                node_data,
                v_node_sizes,
                v_num_placed_nodes,
                p_nodes_status,
                p_node_link_sum_resource,
                p_node_link_max_resource,
                p_node_link_mean_resource,
                self.p_net_node_degrees,
                avg_distance,
            ),
            axis=-1,
        )
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        link_data = self.obs_handler.get_link_attrs_obs(
            self.p_net,
            link_attr_types=attr_type_list,
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        return {'x': node_data, 'edge_index': edge_index, 'edge_attr': link_data}

    def _get_v_net_obs(self):
        attr_type_list = ['resource']
        v_node_sizes = np.ones((self.v_net.num_nodes, 1), dtype=np.float32) * self.v_net.num_nodes / 10
        v_num_placed_nodes = np.ones((self.v_net.num_nodes, 1), dtype=np.float32) * self.num_placed_v_net_nodes / 10
        node_data = self.obs_handler.get_node_attrs_obs(
            self.v_net,
            node_attr_types=attr_type_list,
            node_attr_benchmarks=self.node_attr_benchmarks
        )
        v_node_status = self.obs_handler.get_v_net_nodes_status(
            self.v_net,
            self.solution['node_slots'],
            consist_decision=True
        )
        v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(
            self.v_net,
            link_attr_types=attr_type_list,
            aggr='max',
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(
            self.v_net,
            link_attr_types=attr_type_list,
            aggr='sum',
            link_sum_attr_benchmarks=self.link_sum_attr_benchmarks
        )
        v_node_link_mean_resource = self.obs_handler.get_link_aggr_attrs_obs(
            self.v_net,
            link_attr_types=attr_type_list,
            aggr='mean',
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        node_data = np.concatenate(
            (
                node_data,
                v_node_sizes,
                v_num_placed_nodes,
                v_node_status,
                v_node_link_max_resource,
                v_node_link_sum_resource,
                v_node_link_mean_resource,
            ),
            axis=-1,
        )
        link_data = self.obs_handler.get_link_attrs_obs(
            self.v_net,
            link_attr_types=attr_type_list,
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        return {'x': node_data, 'edge_index': edge_index, 'edge_attr': link_data}

    def compute_reward(self, solution):
        weight = 1 / self.v_net.num_nodes
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = weight
        else:
            reward = -weight
        self.solution['v_net_reward'] += reward
        return reward


@SolverRegistry.register(solver_name='flag_vne', solver_type='r_learning')
class FlagVneSolver(InstanceAgent, PPOSolver):
    flag_vne_gnn_class = GCNConvNet

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        flag_cfg = getattr(config.rl, 'flag_vne', None)
        self.use_bidirectional_action = getattr(flag_cfg, 'use_bidirectional_action', True)
        self.use_meta_learning = getattr(flag_cfg, 'use_meta_learning', True)
        self.num_meta_learning_epochs = getattr(flag_cfg, 'num_meta_learning_epochs', 40)
        self.use_curriculum_scheduling_strategy = getattr(flag_cfg, 'use_curriculum_scheduling_strategy', False)
        self.policy_entropy_threshold = getattr(flag_cfg, 'policy_entropy_threshold', 2.0)
        self.infer_with_meta_policy = getattr(flag_cfg, 'infer_with_meta_policy', False)

        InstanceAgent.__init__(self, FlagVneInstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)

        self.coef_entropy_loss = 0.00
        self.coef_mask_loss = 0.01
        self.clip_grad = True
        self.max_grad_norm = self.config.rl.max_grad_norm
        self.lr_actor = self.config.rl.learning_rate.actor
        self.lr_critic = self.config.rl.learning_rate.critic
        self.lr = self.config.rl.learning_rate.actor
        self.weight_decay = self.config.rl.weight_decay
        self.norm_advantage = self.config.rl.norm_advantage
        self.norm_reward = self.config.rl.norm_reward
        self.eps_clip = self.config.rl.eps_clip
        self.gae_lambda = self.config.rl.gae_lambda
        self.gamma = self.config.rl.gamma
        self.repeat_times = self.config.rl.repeat_times
        self.meta_update_time = 0

        if self.use_meta_learning:
            self.meta_policy = copy.deepcopy(self.policy).to(self.device)
            self.meta_optimizer = torch.optim.AdamW(
                self.meta_policy.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay
            )
            self.task_policies = {}
            self.task_optimizers = {}
            self.target_steps = 516
            self.outer_repeat_times = 1
            self.inner_repeat_times = self.repeat_times
            self.init_inner_kl_penalty = 1e-3
            self.repeat_times = self.repeat_times * 10
            if self.use_curriculum_scheduling_strategy:
                self.training_task_id_list = []
            self.instance_dict = {}

    def solve(self, instance):
        v_net_size = instance['v_net'].num_nodes
        if self.use_meta_learning:
            if self.infer_with_meta_policy or v_net_size not in self.task_policies:
                self.policy = self.meta_policy
            else:
                self.policy = self.task_policies[v_net_size]
        v_net, p_net = instance['v_net'], instance['p_net']
        instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, self.logger, self.config)
        obs = instance_env.get_observation()
        done = False
        while not done:
            tensor_obs = self.preprocess_obs(obs, device=self.device)
            action, _ = self.select_action(tensor_obs, sample=False)
            obs, _, done, _ = instance_env.step(action)
            if done:
                return instance_env.solution
        raise RuntimeError('FlagVNE solve did not finish')

    def select_action(self, observation, sample=True):
        v_net_size = int(observation['v_net_size'].item())
        mask = observation['action_mask'].reshape(
            observation['curr_v_node_id'].shape[0], v_net_size, -1
        ).permute(0, 2, 1)
        if self.use_bidirectional_action:
            high_level_mask = (mask.sum(1) != 0).float()
            with torch.no_grad():
                high_level_action_logits = self.policy.forward(observation, actor_high=True)
            high_level_candidate_action_logits = apply_mask_to_logit(high_level_action_logits, high_level_mask)
            high_level_candidate_action_dist = Categorical(logits=high_level_candidate_action_logits / self.softmax_temp)
            if sample:
                high_level_action = high_level_candidate_action_dist.sample()
            else:
                high_level_action = high_level_candidate_action_logits.argmax(-1)
            high_level_action_logprob = high_level_candidate_action_dist.log_prob(high_level_action)
        else:
            high_level_action = observation['curr_v_node_id']
            high_level_action_logprob = 0

        low_level_mask = mask[torch.arange(mask.shape[0]), :, high_level_action]
        with torch.no_grad():
            low_level_action_logits = self.policy.forward(
                observation, actor_low=True, high_level_action=high_level_action
            )
        low_level_candidate_action_logits = apply_mask_to_logit(low_level_action_logits, low_level_mask)
        low_level_candidate_action_dist = Categorical(logits=low_level_candidate_action_logits / self.softmax_temp)
        if sample:
            low_level_action = low_level_candidate_action_dist.sample()
        else:
            low_level_action = low_level_candidate_action_logits.argmax(-1)
        low_level_action_logprob = low_level_candidate_action_dist.log_prob(low_level_action)

        action_logprob = high_level_action_logprob + low_level_action_logprob
        action = observation['v_net_size'] * low_level_action + high_level_action
        if torch.numel(action) == 1:
            action = action.item()
        else:
            action = action.reshape(-1, ).cpu().detach().numpy()
        return action, action_logprob.cpu().detach().numpy()

    def evaluate_actions(self, old_observations, old_actions, return_others=False):
        v_net_sizes = old_observations['v_net_size']
        high_level_old_actions = (old_actions % v_net_sizes).long()
        low_level_old_actions = (old_actions // v_net_sizes).long()
        max_v_net_size = int(v_net_sizes.max().item())
        mask = old_observations['action_mask'].reshape(v_net_sizes.shape[0], max_v_net_size, -1).permute(0, 2, 1)
        high_level_mask = (mask.sum(1) != 0).float()
        low_level_mask = mask[torch.arange(mask.shape[0]), :, high_level_old_actions]

        if self.use_bidirectional_action:
            high_level_action_logits = self.policy.forward(old_observations, actor_high=True)
            high_level_candidate_action_logits = apply_mask_to_logit(high_level_action_logits, high_level_mask)
            high_level_candidate_action_probs = F.softmax(high_level_candidate_action_logits, dim=-1)
            high_level_policy_dist = Categorical(high_level_candidate_action_probs)
            high_level_action_logprobs = high_level_policy_dist.log_prob(high_level_old_actions)
            high_level_dist_entropy = high_level_policy_dist.entropy()
        else:
            high_level_action_logprobs = 0
            high_level_dist_entropy = 0

        low_level_actions_logits = self.policy.forward(
            old_observations, actor_low=True, high_level_action=high_level_old_actions
        )
        low_level_candidate_actions_logits = apply_mask_to_logit(low_level_actions_logits, low_level_mask)
        low_level_candidate_actions_probs = F.softmax(low_level_candidate_actions_logits, dim=-1)
        low_level_policy_dist = Categorical(low_level_candidate_actions_probs)
        low_level_action_logprobs = low_level_policy_dist.log_prob(low_level_old_actions)
        low_level_dist_entropy = low_level_policy_dist.entropy()

        dist_entropy = high_level_dist_entropy + low_level_dist_entropy
        action_logprobs = high_level_action_logprobs + low_level_action_logprobs
        values = (
            self.policy.forward(old_observations, critic=True).squeeze(-1)
            if hasattr(self.policy, 'evaluate') else None
        )

        if return_others:
            other = {}
            return values, action_logprobs, dist_entropy, other
        return values, action_logprobs, dist_entropy

    def save_model(self, checkpoint_fname):
        if not self.use_meta_learning:
            return super().save_model(checkpoint_fname)
        checkpoint_fname = os.path.join(self.model_dir, checkpoint_fname)
        model_list = [(task_id, self.task_policies[task_id], self.task_optimizers[task_id]) for task_id in self.task_policies.keys()]
        task_model_dict = {}
        task_model_dict['meta_policy'] = {
            'policy': self.meta_policy.state_dict(),
            'optimizer': self.meta_optimizer.state_dict(),
        }
        task_model_dict['task_policies'] = {}
        for task_id, policy, optimizer in model_list:
            task_model_dict['task_policies'][task_id] = {
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        torch.save(task_model_dict, checkpoint_fname)
        if self.verbose >= 0:
            self.logger.info(f'Save model to {checkpoint_fname}')

    def load_model(self, checkpoint_path):
        if not self.use_meta_learning:
            return super().load_model(checkpoint_path)
        if self.verbose >= 0:
            self.logger.info('Attempting to load the pretrained model')
        try:
            checkpoint = torch.load(checkpoint_path)
            self.meta_policy.load_state_dict(checkpoint['meta_policy']['policy'])
            self.meta_optimizer.load_state_dict(checkpoint['meta_policy']['optimizer'])
            if self.verbose >= 0:
                self.logger.info('Loaded pretrained meta policy')
            for task_id in checkpoint['task_policies'].keys():
                if task_id not in self.task_policies:
                    self.task_policies[task_id] = copy.deepcopy(self.meta_policy)
                    self.task_optimizers[task_id] = torch.optim.AdamW(
                        self.task_policies[task_id].parameters(), lr=self.lr_actor, weight_decay=self.weight_decay
                    )
                    self.logger.info(f'New task policy is created for task {task_id}')
                self.task_policies[task_id].load_state_dict(checkpoint['task_policies'][task_id]['policy'])
                self.task_optimizers[task_id].load_state_dict(checkpoint['task_policies'][task_id]['optimizer'])
            if self.verbose >= 0:
                self.logger.info(f'Loaded pretrained model from {checkpoint_path}')
        except Exception:
            if self.verbose >= 0:
                self.logger.info(f'Load failed from {checkpoint_path}\nInitilized with random parameters')

    def learn_with_instance(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        if not self.use_meta_learning:
            return super().learn_with_instance(instance)
        v_net_size = v_net.num_nodes
        task_id = v_net_size
        if task_id not in self.task_policies:
            self._init_task_policy_and_task_optimizer(task_id)
        self.policy = self.task_policies[v_net_size]
        self.optimizer = self.task_optimizers[v_net_size]
        if self.use_meta_learning and self.training_epoch_id < self.num_meta_learning_epochs:
            if v_net_size in self.instance_dict.keys():
                if len(self.instance_dict[v_net_size]) >= 100:
                    self.instance_dict[v_net_size].pop(0)
                self.instance_dict[v_net_size].append(copy.deepcopy(instance))
            else:
                self.instance_dict[v_net_size] = [instance]
        return super().learn_with_instance(instance)

    def collect_new_task_buffer(self, policy, instance_set, max_num_instances=float('inf'), max_num_experiences=float('inf')):
        self.policy = policy
        task_specific_buffer = RolloutBuffer()
        for i, instance in enumerate(instance_set):
            solution, instance_buffer, last_value = super().learn_with_instance(instance)
            self.merge_instance_experience(instance, solution, instance_buffer, last_value)
            instance_buffer.compute_returns_and_advantages(
                last_value, gamma=self.gamma, gae_lambda=self.gae_lambda, method=self.compute_advantage_method
            )
            task_specific_buffer.merge(instance_buffer)
            if task_specific_buffer.size() >= max_num_experiences or i >= max_num_instances:
                break
        return task_specific_buffer

    def update(self):
        if not self.use_meta_learning:
            return super().update()
        if self.training_epoch_id < self.num_meta_learning_epochs:
            if self.verbose >= 0:
                self.logger.info('Meta Learning Process')
            self._meta_learning_update()
        else:
            if self.verbose >= 0:
                self.logger.info('Fine Tuning Process')
            self._fine_tuning_update()

    def _init_task_policy_and_task_optimizer(self, task_id):
        self.task_policies[task_id] = copy.deepcopy(self.meta_policy)
        self.task_optimizers[task_id] = torch.optim.AdamW(
            self.task_policies[task_id].parameters(), lr=self.lr_actor, weight_decay=self.weight_decay
        )
        if self.verbose >= 0:
            self.logger.info(f'New task policy is created for task {task_id}')

    def _stats_task_dist(self, buffer):
        v_net_size_list = np.array([obs['v_net_size'] for obs in buffer.observations])
        counter = Counter(v_net_size_list)
        return dict(sorted(counter.items(), key=lambda x: x[0], reverse=False))

    def _stats_instance_dist(self):
        instance_dist = {task_id: len(self.instance_dict[task_id]) for task_id in self.instance_dict.keys()}
        return dict(sorted(instance_dist.items(), key=lambda x: x[0], reverse=False))

    def _split_buffer(self, buffer):
        task_buffers = {}
        v_net_size_list = np.array([obs['v_net_size'] for obs in buffer.observations])
        tasks_list = sorted(list(set(v_net_size_list)))
        for task_id in tasks_list:
            task_indices = np.where(v_net_size_list == task_id)[0]
            task_buffer = RolloutBuffer()
            task_buffer.observations = [buffer.observations[i] for i in task_indices]
            task_buffer.actions = np.array(buffer.actions)[task_indices].tolist()
            task_buffer.logprobs = np.array(buffer.logprobs)[task_indices].tolist()
            task_buffer.rewards = np.array(buffer.rewards)[task_indices].tolist()
            task_buffer.returns = np.array(buffer.returns)[task_indices].tolist()
            task_buffer.dones = np.array(buffer.dones)[task_indices].tolist()
            task_buffer.values = np.array(buffer.values)[task_indices].tolist()
            task_buffers[task_id] = task_buffer
        return {k: task_buffers[k] for k in sorted(list(task_buffers.keys()))}

    def _meta_learning_update(self):
        import higher
        import torchopt

        self.policy = self.meta_policy
        meta_buffer = self.buffer
        task_buffers = self._split_buffer(meta_buffer)
        task_dist = self._stats_task_dist(meta_buffer)
        instance_dist = self._stats_instance_dist()
        if self.verbose >= 0:
            self.logger.info(f'Experience distribution: {task_dist}')
            self.logger.info(f'Instance distribution: {instance_dist}')
        if self.use_curriculum_scheduling_strategy:
            if len(self.training_task_id_list) == 0:
                self.training_task_id_list.append(min(task_dist.keys()))
                self.logger.info(f'Add task {min(task_dist.keys())} to training task list')
            if self.verbose >= 0:
                self.logger.info(f'Training task list: {self.training_task_id_list}')

        num_tasks = len(task_buffers.keys())
        self.outer_repeat_times = 1
        torch.autograd.set_detect_anomaly(True)
        for _ in range(self.outer_repeat_times):
            kls = []
            total_meta_loss = 0
            total_kl_loss = 0
            total_ppo_loss = 0
            aver_actor_loss = 0
            aver_critic_loss = 0
            aver_entropy_loss = 0
            training_tasks_list = sorted(list(task_buffers.keys()))
            if self.use_curriculum_scheduling_strategy:
                training_tasks_list = self.training_task_id_list
            task_policy_entropy_dict = {}
            inner_opt = torchopt.MetaSGD(self.meta_policy, lr=0.01, weight_decay=self.weight_decay)
            policy_state_dict = torchopt.extract_state_dict(self.meta_policy)
            optim_state_dict = torchopt.extract_state_dict(inner_opt)

            for task_id in training_tasks_list:
                inner_kls = []
                inner_repeat_times = int((num_tasks * self.repeat_times) / len(training_tasks_list)) if self.use_curriculum_scheduling_strategy else self.inner_repeat_times
                inner_repeat_times = 10
                for step in range(inner_repeat_times):
                    buffer = task_buffers[task_id]
                    buffer.split_with_instance()
                    observations, actions, old_action_logprobs, _, returns = self._preprocess_buffer(buffer)
                    loss, (actor_loss, critic_loss, entropy_loss, values, action_logprobs, advantages, kl_div) = \
                        self._calculate_ppo_loss(observations, actions, old_action_logprobs, returns, clip_loss=True)
                    inner_opt.step(loss)
                    inner_kls.append(kl_div)
                    kls.append(kl_div.detach())
                if self.verbose >= 0:
                    self.logger.info(
                        f'Update time: {self.meta_update_time:06d} | '
                        f'task={task_id} step={step} '
                        f'actor_loss {actor_loss.detach():+.4f} '
                        f'critic_loss {critic_loss.detach():+.4f} '
                        f'value {values.detach().mean():+.4f} '
                        f'entropy {entropy_loss.mean():+.4f} '
                        f'advantage {advantages.mean():+.4f}'
                    )
                    self.meta_update_time += 1
                task_specific_buffer = self.collect_new_task_buffer(
                    self.meta_policy, self.instance_dict[task_id], max_num_experiences=64
                )
                observations, actions, old_action_logprobs, _, returns = self._preprocess_buffer(task_specific_buffer)
                ppo_loss, (actor_loss, critic_loss, entropy_loss, values, action_logprobs, advantages, kl_div) = \
                    self._calculate_ppo_loss(observations, actions, old_action_logprobs, returns, clip_loss=True, use_ppo=False)
                kl_loss = torch.zeros(1).to(self.device).mean()
                meta_loss = ppo_loss + kl_loss
                if self.verbose >= 0:
                    self.logger.info(
                        f'Update time: {self.meta_update_time:06d} | '
                        f'task={task_id} '
                        f'meta_loss {loss:+.4f} '
                        f'kl {kl_div:+.4f} '
                        f'actor_loss {actor_loss:+.4f} '
                        f'critic_loss {critic_loss:+.4f} '
                        f'entropy {entropy_loss:+.4f}'
                    )
                    self.meta_update_time += 1
                meta_loss.backward()
                torchopt.recover_state_dict(self.meta_policy, policy_state_dict)
                torchopt.recover_state_dict(inner_opt, optim_state_dict)
                total_meta_loss += meta_loss.detach().cpu().numpy()
                total_kl_loss += kl_loss.detach().cpu().numpy()
                total_ppo_loss += ppo_loss.detach().cpu().numpy()
                aver_entropy_loss += entropy_loss.detach().cpu().numpy()
                aver_actor_loss += actor_loss.detach().cpu().numpy()
                aver_critic_loss += critic_loss.detach().cpu().numpy()
                task_policy_entropy_dict[task_id] = entropy_loss.detach().cpu().numpy()

            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), self.max_grad_norm)
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()
            for _, param in self.meta_policy.named_parameters():
                if torch.isnan(param).any():
                    raise RuntimeError('NaN found in meta policy parameters')

            num_tasks = len(training_tasks_list)
            aver_actor_loss /= num_tasks
            aver_critic_loss /= num_tasks
            aver_entropy_loss /= num_tasks

            if self.use_curriculum_scheduling_strategy:
                most_complex_task_id = self.training_task_id_list[-1]
                if most_complex_task_id < max(task_dist.keys()):
                    most_complex_task_policy_entropy = task_policy_entropy_dict[most_complex_task_id]
                    if most_complex_task_policy_entropy < self.policy_entropy_threshold:
                        self.training_task_id_list.append(most_complex_task_id + 1)
                        self.logger.info(f'Add task {most_complex_task_id + 1} to training task list')

            if self.verbose >= 0:
                self.logger.info(
                    f'Update time: {self.meta_update_time:06d} | '
                    f'meta_loss {total_meta_loss:+.4f} '
                    f'ppo_loss {total_ppo_loss:+.4f} '
                    f'actor_loss {aver_actor_loss:+.4f} '
                    f'critic_loss {aver_critic_loss:+.4f} '
                    f'entropy {aver_entropy_loss:+.4f} '
                    f'kl_loss {total_kl_loss:+.4f}'
                )
                self.meta_update_time += 1

        for task_id in task_buffers.keys():
            self.task_policies[task_id].load_state_dict(self.meta_policy.state_dict())
        meta_buffer.clear()
        self.buffer = meta_buffer
        self.instance_dict = {}

    def _fine_tuning_update(self):
        meta_buffer = self.buffer
        task_dist = self._stats_task_dist(self.buffer)
        if self.verbose >= 0:
            self.logger.info(f'Task distribution: {task_dist}')
        for task_id in task_dist.keys():
            if task_id not in self.task_policies:
                self.task_policies[task_id] = copy.deepcopy(self.meta_policy)
                self.task_optimizers[task_id] = torch.optim.AdamW(
                    self.task_policies[task_id].parameters(), lr=self.lr
                )
        task_buffers = self._split_buffer(self.buffer)
        task_ids = sorted(list(task_buffers.keys()))
        for task_id in task_ids:
            self.policy = self.task_policies[task_id]
            self.optimizer = self.task_optimizers[task_id]
            self.buffer = task_buffers[task_id]
            super().update()
        meta_buffer.clear()

    def _calculate_ppo_loss(self, observations, actions, old_action_logprobs, returns, clip_loss=False, use_ppo=True):
        values, action_logprobs, dist_entropy = self.evaluate_actions(observations, actions)
        advantages = returns - values.detach()
        if self.norm_advantage and values.numel() != 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        if use_ppo:
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() if clip_loss else -(surr1).mean()
            critic_loss = self.criterion_critic(returns, values)
            kl_div = ((torch.exp(ratio) - 1) - ratio).mean()
        else:
            actor_loss = -(action_logprobs * advantages).mean()
            kl_div = torch.zeros(1).to(self.device).mean()
        critic_loss = self.criterion_critic(returns, values)
        entropy_loss = dist_entropy.mean()
        loss = actor_loss + self.coef_critic_loss * critic_loss - self.coef_entropy_loss * entropy_loss
        return loss, (actor_loss, critic_loss, entropy_loss, values, action_logprobs, advantages, kl_div)

    def _preprocess_buffer(self, buffer):
        batch_observations = self.preprocess_obs(buffer.observations, self.device)
        batch_actions = torch.LongTensor(np.array(buffer.actions)).to(self.device)
        batch_old_action_logprobs = torch.FloatTensor(np.concatenate(buffer.logprobs, axis=0)).to(self.device)
        batch_rewards = torch.FloatTensor(buffer.rewards).to(self.device)
        batch_returns = torch.FloatTensor(buffer.returns).to(self.device)
        if self.norm_reward:
            batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-9)
        return batch_observations, batch_actions, batch_old_action_logprobs, batch_rewards, batch_returns


@SolverRegistry.register(solver_name='flag_vne_meta_free_multi_policy', solver_type='r_learning')
class FlagVneMetaFreeMultiPolicySolver(FlagVneSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.use_bidirectional_action = True
        self.use_meta_learning = True
        self.num_meta_learning_epochs = 0
        self.use_curriculum_scheduling_strategy = False
        self.infer_with_meta_policy = False


@SolverRegistry.register(solver_name='flag_vne_meta_free_single_policy', solver_type='r_learning')
class FlagVneMetaFreeSinglePolicySolver(FlagVneSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.use_bidirectional_action = True
        self.use_meta_learning = False
        self.num_meta_learning_epochs = 0
        self.use_curriculum_scheduling_strategy = False
        self.infer_with_meta_policy = False


@SolverRegistry.register(solver_name='flag_vne_meta_policy', solver_type='r_learning')
class FlagVneMetaPolicySolver(FlagVneSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.use_bidirectional_action = True
        self.use_meta_learning = False
        self.num_meta_learning_epochs = 0
        self.use_curriculum_scheduling_strategy = False
        self.infer_with_meta_policy = True


@SolverRegistry.register(solver_name='flag_vne_no_curriculum', solver_type='r_learning')
class FlagVneNoCurriculumSolver(FlagVneSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.use_bidirectional_action = True
        self.use_meta_learning = False
        self.num_meta_learning_epochs = 0
        self.use_curriculum_scheduling_strategy = True
        self.infer_with_meta_policy = False


@SolverRegistry.register(solver_name='flag_vne_unidirectional_action', solver_type='r_learning')
class FlagVneUnidirectionalActionSolver(FlagVneSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)
        self.use_bidirectional_action = False
        self.use_meta_learning = False
        self.num_meta_learning_epochs = 0
        self.use_curriculum_scheduling_strategy = False
        self.infer_with_meta_policy = False


@SolverRegistry.register(solver_name='ppo_bigcn+', solver_type='r_learning')
class PpoBigCNSolver(FlagVneSolver):
    flag_vne_gnn_class = GCNConvNet


@SolverRegistry.register(solver_name='ppo_bigat+', solver_type='r_learning')
class PpoBigATSolver(FlagVneSolver):
    flag_vne_gnn_class = GATConvNet


def make_policy(agent, **kwargs):
    num_vn_attrs = agent.config.simulation.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.config.simulation.v_sim_setting_num_link_resource_attrs
    policy = FlagVneActorCritic(
        p_net_num_nodes=agent.config.simulation.p_net_setting_num_nodes,
        p_net_feature_dim=num_vn_attrs + num_vl_attrs * 3 + 5,
        p_net_edge_dim=num_vl_attrs,
        v_net_feature_dim=num_vn_attrs + num_vl_attrs * 3 + 3,
        v_net_edge_dim=num_vl_attrs,
        embedding_dim=agent.config.nn.embedding_dim,
        dropout_prob=agent.config.nn.dropout_prob,
        batch_norm=agent.config.nn.batch_norm,
        gnn_class=getattr(agent, 'flag_vne_gnn_class', GCNConvNet),
    ).to(agent.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=agent.config.rl.learning_rate.actor, weight_decay=agent.config.rl.weight_decay)
    return policy, optimizer


def obs_as_tensor(obs, device):
    if isinstance(obs, dict):
        p_net_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
        v_net_data = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs['v_net_edge_attr'])
        obs_p_net = Batch.from_data_list([p_net_data]).to(device)
        obs_v_net = Batch.from_data_list([v_net_data]).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array([obs['curr_v_node_id']])).to(device)
        obs_action_mask = torch.FloatTensor(np.array([obs['action_mask']])).to(device)
        obs_v_net_size = torch.LongTensor(np.array([obs['v_net_size']])).to(device)
        return {
            'p_net': obs_p_net,
            'v_net': obs_v_net,
            'curr_v_node_id': obs_curr_v_node_id,
            'action_mask': obs_action_mask,
            'v_net_size': obs_v_net_size,
        }
    if isinstance(obs, list):
        p_net_data_list, v_net_data_list, curr_v_node_id_list, action_mask_list, v_net_size_list = [], [], [], [], []
        for observation in obs:
            p_net_data_list.append(
                get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
            )
            v_net_data_list.append(
                get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
            )
            curr_v_node_id_list.append(observation['curr_v_node_id'])
            action_mask_list.append(observation['action_mask'])
            v_net_size_list.append(observation['v_net_size'])
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        obs_v_net = Batch.from_data_list(v_net_data_list).to(device)
        obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
        max_len_action_mask = max(len(seq) for seq in action_mask_list)
        padded_action_mask = np.zeros((len(action_mask_list), max_len_action_mask))
        for i, seq in enumerate(action_mask_list):
            padded_action_mask[i, :len(seq)] = seq
        obs_action_mask = torch.FloatTensor(np.array(padded_action_mask)).to(device)
        return {
            'p_net': obs_p_net,
            'v_net': obs_v_net,
            'curr_v_node_id': obs_curr_v_node_id,
            'action_mask': obs_action_mask,
            'v_net_size': obs_v_net_size,
        }
    raise ValueError(f'Unrecognized type of observation {type(obs)}')
