import math
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from virne.solver.learning.neural_network import GraphPooling, GCNConvNet, GATConvNet


class FlagVneActorCritic(nn.Module):

    def __init__(
        self,
        p_net_num_nodes,
        p_net_feature_dim,
        p_net_edge_dim,
        v_net_feature_dim,
        v_net_edge_dim,
        embedding_dim=128,
        dropout_prob=0.0,
        batch_norm=False,
        gnn_class=GCNConvNet,
    ):
        super().__init__()
        self.encoder = FlagVneBaseModel(
            p_net_num_nodes,
            p_net_feature_dim,
            p_net_edge_dim,
            v_net_feature_dim,
            v_net_edge_dim,
            embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            gnn_class=gnn_class,
        )
        self.actor = FlagVneActor(
            p_net_num_nodes,
            p_net_feature_dim,
            p_net_edge_dim,
            v_net_feature_dim,
            v_net_edge_dim,
            embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        )
        self.critic = FlagVneCritic(
            p_net_num_nodes,
            p_net_feature_dim,
            p_net_edge_dim,
            v_net_feature_dim,
            v_net_edge_dim,
            embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        )

    def forward(self, x, actor_high=False, actor_low=False, critic=False, high_level_action=None):
        v_g_emb, v_node_dense, _, p_g_emb, p_node_dense, _ = self.encoder(x)
        if actor_high:
            state_embeddings = v_node_dense + p_g_emb.unsqueeze(1)
            return self.actor.get_high_policy(state_embeddings)
        if actor_low:
            if high_level_action is None:
                high_level_action = x['curr_v_node_id']
            curr_v_node_id = high_level_action.unsqueeze(1).unsqueeze(1).long()
            curr_v_node_emb = v_node_dense.gather(
                1,
                curr_v_node_id.expand(v_node_dense.size(0), -1, v_node_dense.size(-1))
            ).squeeze(1)
            v_g_emb = v_g_emb + curr_v_node_emb
            state_embeddings = p_node_dense + v_g_emb.unsqueeze(1)
            return self.actor.get_low_policy(state_embeddings)
        if critic:
            p_node_dense = p_node_dense + v_g_emb.unsqueeze(1)
            return self.critic(p_node_dense)
        return None

    def evaluate(self, x):
        v_g_emb, v_node_dense, _, p_g_emb, p_node_dense, _ = self.encoder(x)
        p_node_dense = p_node_dense + v_g_emb.unsqueeze(1)
        return self.critic(p_node_dense)


class FlagVneActor(nn.Module):

    def __init__(
        self,
        p_net_num_nodes,
        p_net_feature_dim,
        p_net_edge_dim,
        v_net_feature_dim,
        v_net_edge_dim,
        embedding_dim=128,
        dropout_prob=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.high_policy = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.low_policy = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def get_high_policy(self, state_embeddings):
        return self.high_policy(state_embeddings).squeeze(-1)

    def get_low_policy(self, state_embeddings):
        return self.low_policy(state_embeddings).squeeze(-1)


class FlagVneCritic(nn.Module):

    def __init__(
        self,
        p_net_num_nodes,
        p_net_feature_dim,
        p_net_edge_dim,
        v_net_feature_dim,
        v_net_edge_dim,
        embedding_dim=128,
        dropout_prob=0.0,
        batch_norm=False,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(p_net_num_nodes, p_net_num_nodes * 2),
            nn.ReLU(),
            nn.Linear(p_net_num_nodes * 2, p_net_num_nodes),
            nn.ReLU(),
            nn.Linear(p_net_num_nodes, 1),
        )

    def forward(self, state_embeddings):
        fusion = self.head(state_embeddings).squeeze(-1)
        return self.value_head(fusion).squeeze(-1)


class FlagVneNetEncoder(nn.Module):

    def __init__(self, net_feature_dim, net_edge_dim, embedding_dim=128, dropout_prob=0.0, batch_norm=False, gnn_class=GCNConvNet):
        super().__init__()
        self.init_lin = nn.Linear(net_feature_dim, embedding_dim)
        self.net_gnn = gnn_class(
            embedding_dim,
            embedding_dim,
            num_layers=2,
            embedding_dim=embedding_dim,
            edge_dim=net_edge_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        )
        self.net_mean_pooling = GraphPooling('mean')

    def forward(self, net_batch):
        node_init_embeddings = self.init_lin(net_batch.x)
        net_batch = net_batch.clone()
        net_batch.x = node_init_embeddings
        node_embeddings = self.net_gnn(net_batch)
        graph_embedding = self.net_mean_pooling(node_embeddings, net_batch.batch)
        node_dense_embeddings, net_mask = to_dense_batch(node_embeddings, net_batch.batch)
        node_init_dense_embeddings, _ = to_dense_batch(node_init_embeddings, net_batch.batch)
        node_dense_embeddings_with_graph_embedding = (
            node_dense_embeddings + graph_embedding.unsqueeze(1) + node_init_dense_embeddings
        )
        return node_embeddings, graph_embedding, node_dense_embeddings_with_graph_embedding, net_mask


class FlagVneBaseModel(nn.Module):

    def __init__(
        self,
        p_net_num_nodes,
        p_net_feature_dim,
        p_net_edge_dim,
        v_net_feature_dim,
        v_net_edge_dim,
        embedding_dim=128,
        dropout_prob=0.0,
        batch_norm=False,
        gnn_class=GCNConvNet,
    ):
        super().__init__()
        self.v_net_encoder = FlagVneNetEncoder(
            v_net_feature_dim,
            v_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            gnn_class=gnn_class,
        )
        self.p_net_encoder = FlagVneNetEncoder(
            p_net_feature_dim,
            p_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            gnn_class=gnn_class,
        )
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                stdv = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, obs):
        v_net_batch, p_net_batch = obs['v_net'], obs['p_net']
        v_node_emb, v_g_emb, v_node_dense, v_net_mask = self.v_net_encoder(v_net_batch)
        p_node_emb, p_g_emb, p_node_dense, p_net_mask = self.p_net_encoder(p_net_batch)
        return v_g_emb, v_node_dense, v_net_mask, p_g_emb, p_node_dense, p_net_mask

