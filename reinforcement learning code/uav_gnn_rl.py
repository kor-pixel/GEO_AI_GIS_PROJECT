# =============================
# 1. DQN NETWORK (MLP)
# =============================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import os
import pandas as pd

from torch_geometric.nn import NNConv
import torch.nn.functional as F


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class GraphDuelingDQN(nn.Module):
    """
    Graph-based Dueling DQN with NoisyNet, dùng PyG NNConv.

    Giả định:
      - Graph cố định cho 1 env (1 time_slot):
          node_feats: (N, F_node)
          edge_index: (2, E)
          edge_attr: (E, F_edge)
      - RL state: [one_hot(node_idx), 4 scalar toàn cục]
          => state_dim = num_nodes + global_dim (global_dim = 4)
      - forward(state) trả về Q(s, a_j) cho mọi action j = 0..N-1
    """

    def __init__(
        self,
        node_feats: torch.Tensor,      # (N, F_node)
        edge_index: torch.Tensor,      # (2, E) long
        edge_attr: torch.Tensor,       # (E, F_edge)
        state_dim: int,                # = num_nodes + global_dim
        global_dim: int = 4,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        device = torch.device(device)

        # ---- Lưu graph vào buffer (không trainable) ----
        node_feats = node_feats.to(device)
        edge_index = edge_index.long().to(device)
        edge_attr = edge_attr.to(device)

        self.register_buffer("node_feats", node_feats)    # (N, F_node)
        self.register_buffer("edge_index", edge_index)    # (2, E)
        self.register_buffer("edge_attr", edge_attr)      # (E, F_edge)

        self.num_nodes = node_feats.size(0)
        self.node_in_dim = node_feats.size(1)
        self.edge_in_dim = edge_attr.size(1)
        self.state_dim = state_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim

        assert state_dim == self.num_nodes + global_dim, (
            f"state_dim phải = num_nodes + global_dim, "
            f"nhưng state_dim={state_dim}, num_nodes={self.num_nodes}, global_dim={global_dim}"
        )

        # --------- GNN layers với PyG NNConv ---------
        # Layer 1: node_in_dim -> hidden_dim
        edge_nn1 = nn.Sequential(
            nn.Linear(self.edge_in_dim, self.edge_in_dim * 2),
            nn.ReLU(),
            nn.Linear(self.edge_in_dim * 2, self.node_in_dim * hidden_dim),
        )
        conv1 = NNConv(
            in_channels=self.node_in_dim,
            out_channels=hidden_dim,
            nn=edge_nn1,
            aggr="add",
        )

        gnn_layers = [conv1]
        # Các layer tiếp theo: hidden_dim -> hidden_dim
        for _ in range(num_gnn_layers - 1):
            edge_nn_k = nn.Sequential(
                nn.Linear(self.edge_in_dim, self.edge_in_dim * 2),
                nn.ReLU(),
                nn.Linear(self.edge_in_dim * 2, hidden_dim * hidden_dim),
            )
            conv_k = NNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                nn=edge_nn_k,
                aggr="add",
            )
            gnn_layers.append(conv_k)

        self.gnn_layers = nn.ModuleList(gnn_layers)

        # --------- Encode global scalar (4 giá trị) -> hidden_dim ---------
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
        )

        # --------- Dueling head (NoisyNet) ---------
        # Value: V(s) từ [h_curr, g]  => (B, 1)
        # v_input có shape (B, 2H), nên in_features = 2*hidden_dim
        self.value_fc = nn.Linear(2 * hidden_dim, hidden_dim)
        self.value_noisy = NoisyLinear(hidden_dim, 1)

        # Advantage: A(s, a_j) từ [h_curr, h_j, g] => scalar cho từng node j
        # adv_input có shape (B, N, 3H), nên in_features = 3*hidden_dim
        self.adv_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        self.adv_noisy = NoisyLinear(hidden_dim, 1)  # per node → 1 scalar, sẽ reshape về (B, N)

    def reset_noise(self):
        # reset noise cho tất cả NoisyLinear
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        Trả về: Q_values: (B, num_nodes)
        """
        device = state.device
        B = state.size(0)

        # 1) Tách state:
        #    - one_hot: (B, N) cho node hiện tại
        #    - global_feats: (B, global_dim)
        one_hot = state[:, : self.num_nodes]            # (B, N)
        global_feats = state[:, self.num_nodes :]       # (B, global_dim)

        # index node hiện tại
        curr_idx = one_hot.argmax(dim=1)                # (B,)

        # 2) GNN trên graph (giống nhau cho cả batch)
        x = self.node_feats                              # (N, F_node)
        edge_index = self.edge_index
        edge_attr = self.edge_attr

        for conv in self.gnn_layers:
            x = conv(x, edge_index, edge_attr)          # (N, hidden_dim)
            x = F.relu(x)

        # node embedding cuối: x (N, H)
        # expand cho batch: (B, N, H)
        x_exp = x.unsqueeze(0).expand(B, -1, -1)        # (B, N, H)

        # embedding của node hiện tại: h_curr (B, H)
        batch_idx = torch.arange(B, device=device)
        h_curr = x_exp[batch_idx, curr_idx]             # (B, H)

        # 3) Global scalar -> g (B, H)
        g = self.global_mlp(global_feats)               # (B, H)

        # 4) Value stream: V(s)
        v_input = torch.cat([h_curr, g], dim=-1)        # (B, 2H)
        v_hidden = F.relu(self.value_fc(v_input))       # (B, H)
        V = self.value_noisy(v_hidden)                  # (B, 1)

        # 5) Advantage stream: A(s, a_j)
        # lặp h_curr và g trên chiều node:
        h_curr_expand = h_curr.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (B, N, H)
        g_expand = g.unsqueeze(1).expand(-1, self.num_nodes, -1)            # (B, N, H)

        # concat [h_curr, h_j, g] cho từng node j
        adv_input = torch.cat([h_curr_expand, x_exp, g_expand], dim=-1)     # (B, N, 3H)

        adv_hidden = F.relu(self.adv_fc(adv_input))      # (B, N, H)

        # flatten B,N để apply NoisyLinear(H → 1)
        adv_hidden_flat = adv_hidden.view(B * self.num_nodes, self.hidden_dim)   # (B*N, H)
        A_flat = self.adv_noisy(adv_hidden_flat)                                 # (B*N, 1)
        A = A_flat.view(B, self.num_nodes)                                       # (B, N)

        # 6) Dueling: Q(s,a) = V(s) + (A - mean_a A)
        A_mean = A.mean(dim=1, keepdim=True)            # (B, 1)
        Q = V + (A - A_mean)                            # broadcasting (B, N)

        return Q


# =============================
class NoisyLinear(nn.Module):
    """
    Noisy linear layer (Fortunato et al., 2017)
    - W = μ_w + σ_w ⊙ ε_w
    - b = μ_b + σ_b ⊙ ε_b
    - ε is resampled every time reset_noise() is called
    """

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Parameters "mean" and "sigma"
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffer (not learnable)
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize according to NoisyNet proposal
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # in eval, can use only μ (or keep noise as you prefer)
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN with NoisyNet:
      - shared feature: FC + ReLU
      - value stream: NoisyLinear
      - advantage stream: NoisyLinear
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_noisy = NoisyLinear(hidden_dim, 1)

        self.adv_fc = nn.Linear(hidden_dim, hidden_dim)
        self.adv_noisy = NoisyLinear(hidden_dim, action_dim)

    def forward(self, x):
        f = self.feature(x)
        v = torch.relu(self.value_fc(f))
        v = self.value_noisy(v)

        a = torch.relu(self.adv_fc(f))
        a = self.adv_noisy(a)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q

    def reset_noise(self):
        # Reset noise for all NoisyLinear layers in the network
        self.value_noisy.reset_noise()
        self.adv_noisy.reset_noise()




# =============================
# 2. REPLAY BUFFER
# =============================

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay 
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        """Add a new transition with priority = current max_priority (or 1.0 if buffer is empty)."""
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Return: batch Transition, indices, weights (tensor)."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        # P(i) ~ p_i^alpha
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            # fallback if all prios = 0
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize max = 1

        batch = Transition(*zip(*samples))
        weights_t = torch.tensor(weights, dtype=torch.float32)

        return batch, indices, weights_t

    def update_priorities(self, indices, priorities):
        """Update priorities for samples used in a batch."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)


# =============================
# 3. ENVIRONMENT FOR ROUTING
# =============================

class UAVRoutingEnvTorch:
    def __init__(
        self,
        weighted_graph: dict,
        max_hops: int,
        start_id: str,
        goal_id: str,
        node_positions: dict,          # node_id -> (lat, lon)
        target_bw_mbps: float = 5.0,
        max_delay_ms: float = 400.0,
        max_jitter_ms: float = 30.0,
        max_loss: float = 0.01,
        # weights cho reward
        w_bw: float = 1.0,
        w_delay: float = 1.0,
        w_jitter: float = 0.5,
        w_loss: float = 2.0,
        w_hop: float = 0.2,
        w_progress: float = 1.0,       # reward/penalty for moving closer/farther from the goal
        goal_bonus: float = 50.0,
        fail_penalty: float = -30.0,
    ):
        if start_id not in weighted_graph:
            raise ValueError(f"start_id={start_id} not in weighted_graph")
        if goal_id not in weighted_graph:
            raise ValueError(f"goal_id={goal_id} not in weighted_graph")

        self.weighted_graph = weighted_graph
        self.max_hops = max_hops
        self.start_id = start_id
        self.goal_id = goal_id

        # QoS scale
        self.target_bw_mbps = target_bw_mbps
        self.max_delay_ms = max_delay_ms
        self.max_jitter_ms = max_jitter_ms
        self.max_loss = max_loss

        # reward weights
        self.w_bw = w_bw
        self.w_delay = w_delay
        self.w_jitter = w_jitter
        self.w_loss = w_loss
        self.w_hop = w_hop
        self.w_progress = w_progress
        self.goal_bonus = goal_bonus
        self.fail_penalty = fail_penalty

        # mapping node_id <-> index
        self.node_ids = sorted(list(weighted_graph.keys()))
        self.id2idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.idx2id = {i: nid for nid, i in self.id2idx.items()}
        self.num_nodes = len(self.node_ids)

        if start_id not in self.id2idx or goal_id not in self.id2idx:
            raise ValueError("start_id or goal_id not in node_ids of weighted_graph")

        self.start_idx = self.id2idx[start_id]
        self.goal_idx = self.id2idx[goal_id]

        # node coordinates: (lat, lon) -> can be seen as (y, x)
        self.coords = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for nid, idx in self.id2idx.items():
            if nid not in node_positions:
                raise ValueError(f"node_positions does not have coordinates for node_id={nid}")
            lat, lon = node_positions[nid]
            self.coords[idx, 0] = lat
            self.coords[idx, 1] = lon

        # Euclidean distance to goal for each node
        goal_lat, goal_lon = self.coords[self.goal_idx]
        dist_list = []
        for i in range(self.num_nodes):
            lat, lon = self.coords[i]
            dx = goal_lat - lat
            dy = goal_lon - lon
            dist = (dx ** 2 + dy ** 2) ** 0.5
            dist_list.append(dist)
        self.dist_to_goal = np.array(dist_list, dtype=np.float32)
        self.max_dist_to_goal = float(self.dist_to_goal.max()) if self.dist_to_goal.size > 0 else 1.0

        # build edge_dict và neighbors_idx
        self.edge_dict = {}
        self.neighbors_idx = {self.id2idx[u_id]: [] for u_id in self.node_ids}
        for u_id, nbrs in weighted_graph.items():
            u_idx = self.id2idx[u_id]
            for (v_id, dist, delay_ms, bw, jitter_ms, loss) in nbrs:
                v_idx = self.id2idx[v_id]
                self.edge_dict[(u_idx, v_idx)] = (dist, delay_ms, bw, jitter_ms, loss)
                self.neighbors_idx[u_idx].append(v_idx)

        # State: one-hot(node_idx) + hops_norm + cum_delay_norm + e2e_loss_norm + dist_to_goal_norm
        self.state_dim = self.num_nodes + 4
        self.action_dim = self.num_nodes

        # internal variables
        self.current_idx: int | None = None
        self.hops_used: int = 0
        self.cum_delay: float = 0.0
        self.cum_jitter: float = 0.0
        self.success_prob: float = 1.0  # P_success = ∏ (1 - loss_i)

    # -------------------------
    # STATE ENCODING
    # -------------------------

    def _encode_state(self):
        """
        [one_hot(node_idx),
         hops_used / max_hops,
         cum_delay / max_delay_ms,
         e2e_loss / max_loss,
         dist_to_goal / max_dist_to_goal]
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[self.current_idx] = 1.0

        state[self.num_nodes + 0] = self.hops_used / max(1, self.max_hops if self.max_hops is not None else 1)
        state[self.num_nodes + 1] = self.cum_delay / max(self.max_delay_ms if self.max_delay_ms is not None else 1e-6, 1e-6)

        e2e_loss = 1.0 - self.success_prob
        if self.max_loss > 0:
            e2e_loss_norm = min(e2e_loss / self.max_loss, 1.0)
        else:
            e2e_loss_norm = e2e_loss
        state[self.num_nodes + 2] = e2e_loss_norm

        d = self.dist_to_goal[self.current_idx]
        d_norm = d / max(self.max_dist_to_goal, 1e-6)
        state[self.num_nodes + 3] = d_norm

        return state

    # -------------------------
    # GYM-LIKE API
    # -------------------------

    def reset(self):
        self.current_idx = self.start_idx
        self.hops_used = 0
        self.cum_delay = 0.0
        self.cum_jitter = 0.0
        self.success_prob = 1.0
        return self._encode_state()

    def step(self, action_idx: int):
        done = False

        if self.current_idx == self.goal_idx:
            # already at goal, should terminate
            return self._encode_state(), 0.0, True, {"info": "already_at_goal"}

        key = (self.current_idx, action_idx)
        if key not in self.edge_dict:
            # invalid move
            reward = self.fail_penalty
            done = True
            info = {"info": "invalid_move"}
            return self._encode_state(), reward, done, info

        # distance to goal before moving
        dist_prev_goal = self.dist_to_goal[self.current_idx]

        # apply move
        dist, delay_ms, bw, jitter_ms, loss = self.edge_dict[key]

        self.current_idx = action_idx
        self.hops_used += 1
        self.cum_delay += delay_ms
        self.cum_jitter += jitter_ms
        self.success_prob *= (1.0 - loss)
        e2e_loss = 1.0 - self.success_prob

        # normalize QoS
        bw_norm = bw / max(self.target_bw_mbps, 1e-6)
        bw_norm = min(bw_norm, 2.0)

        hop_delay_norm = delay_ms / max(self.max_delay_ms, 1e-6)
        hop_jitter_norm = jitter_ms / max(self.max_jitter_ms, 1e-6)
        e2e_loss_norm = e2e_loss / max(self.max_loss, 1e-6) if self.max_loss > 0 else e2e_loss
        e2e_loss_norm = min(max(e2e_loss_norm, 0.0), 2.0)

        # QoS-based step reward
        reward_step = (
            + self.w_bw     * bw_norm
            - self.w_delay  * hop_delay_norm
            - self.w_jitter * hop_jitter_norm
            - self.w_loss   * e2e_loss_norm
            - self.w_hop
        )

        # progress shaping: if moving closer to the goal, reward the agent
        dist_next_goal = self.dist_to_goal[self.current_idx]
        delta_d = dist_prev_goal - dist_next_goal  # >0 if closer
        delta_d_norm = delta_d / max(self.max_dist_to_goal, 1e-6)
        reward_progress = self.w_progress * delta_d_norm

        reward = reward_step + reward_progress

        info = {
            "dist": dist,
            "delay_ms": delay_ms,
            "bw": bw,
            "jitter_ms": jitter_ms,
            "hop": self.hops_used,
            "e2e_loss": e2e_loss,
            "cum_delay": self.cum_delay,
            "cum_jitter": self.cum_jitter,
        }

        # ----------------------
        # Termination checks
        # ----------------------
        avg_jitter = self.cum_jitter / max(self.hops_used, 1)

        if (self.max_loss is not None) and (e2e_loss > self.max_loss):
            reward += self.fail_penalty
            done = True
            info["info"] = "e2e_loss_exceeded"

        elif self.cum_delay > self.max_delay_ms:
            reward += self.fail_penalty
            done = True
            info["info"] = "delay_exceeded"

        elif avg_jitter > self.max_jitter_ms:
            reward += self.fail_penalty
            done = True
            info["info"] = "jitter_exceeded"

        elif self.hops_used >= self.max_hops and self.current_idx != self.goal_idx:
            reward += self.fail_penalty
            done = True
            info["info"] = "max_hops_exceeded"

        elif self.current_idx == self.goal_idx:
            # check QoS end-to-end
            qos_ok = (
                (self.cum_delay <= self.max_delay_ms)
                and (e2e_loss <= self.max_loss)
                and (avg_jitter <= self.max_jitter_ms)
            )

            if qos_ok:
                reward += self.goal_bonus
                # more remaining hops -> more reward
                reward += 5.0 * max(self.max_hops - self.hops_used, 0)
                info["info"] = "reached_goal_qos_ok"
            else:
                reward += -10.0
                info["info"] = "reached_goal_qos_bad"

            done = True

        else:
            info["info"] = "step"

        next_state = self._encode_state()
        return next_state, reward, done, info

    # -------------------------
    # INFERENCE
    # -------------------------

    def get_valid_actions(self, node_idx=None):
        if node_idx is None:
            node_idx = self.current_idx
        if node_idx is None:
            return []
        return self.neighbors_idx.get(node_idx, [])

    def decode_path(self, indices):
        return [self.idx2id[i] for i in indices]


# =============================
# 4. DQN AGENT
# =============================

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 500,
        device: str = "cpu",

        # --- PER params ---
        use_per: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_eps: float = 1e-5,

        epsilon_start: float = 0.0,
        epsilon_min: float = 0.0,
        epsilon_decay: float = 1.0,
        
        use_gnn: bool = False,
        graph_tensors=None,
        global_dim: int = 4,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.use_gnn = use_gnn
        
        if use_gnn:
            assert graph_tensors is not None, "graph_tensors must be provided for GNN"
            node_feats, edge_index, edge_attr = graph_tensors
        
            self.policy_net = GraphDuelingDQN(
                    node_feats=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    state_dim=state_dim,
                    global_dim=global_dim,
                    hidden_dim=hidden_dim,
                    num_gnn_layers=num_gnn_layers,
                    device=device,
                ).to(device)
            
            self.target_net = GraphDuelingDQN(
                    node_feats=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    state_dim=state_dim,
                    global_dim=global_dim,
                    hidden_dim=hidden_dim,
                    num_gnn_layers=num_gnn_layers,
                    device=device,
                ).to(device)
        else:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim=hidden_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        # self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.target_update_freq = target_update_freq
        self.device = device

        self.train_steps = 0

        # ----------------
        # PER / Replay
        # ----------------
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta_start
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.per_eps = per_eps

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha
            )
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.train_steps = 0

        self.use_gnn = use_gnn

    def select_action(self, state, valid_actions=None, explore: bool = True):
        """

        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, state_dim) hoặc cái gì đó tuỳ policy_net

        # Exploration với NoisyNet
        if explore and hasattr(self.policy_net, "reset_noise") and self.policy_net.training:
            self.policy_net.reset_noise()

        with torch.no_grad():
            q_values = self.policy_net(state_t)  # shape gì đó (B, A) hoặc (A,)...

        # ÉP về vector 1D: (action_dim,)
        q_values = q_values.view(-1)

        # Nếu không truyền valid_actions -> chọn argmax toàn bộ
        if valid_actions is None or len(valid_actions) == 0:
            action = int(torch.argmax(q_values).item())
            return action

        # Đảm bảo valid_actions là list các int
        va = list(valid_actions)

        # Dùng tensor index để lấy q_sub
        va_tensor = torch.tensor(va, dtype=torch.long, device=self.device)
        q_sub = q_values[va_tensor]  # (len(va),)

        # Argmax trên tập này -> index nội bộ trong va
        idx_in_va = int(torch.argmax(q_sub).item())

        # Phòng hộ: nếu có bug shape thì vẫn không crash
        if idx_in_va < 0 or idx_in_va >= len(va):
            # fallback: clamp về biên
            idx_in_va = max(0, min(len(va) - 1, idx_in_va))

        action = va[idx_in_va]
        return action



    def store_transition(self, *args):
        self.buffer.push(*args)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        # =============================
        # 1) get batch from buffer
        # =============================
        if self.use_per:
            batch, indices, weights = self.buffer.sample(
                self.batch_size, beta=self.per_beta
            )
            weights = weights.to(self.device).unsqueeze(1)  # (B,1)
        else:
            batch = self.buffer.sample(self.batch_size)
            indices = None
            weights = None

        state_batch = torch.tensor(
            np.array(batch.state),
            dtype=torch.float32,
            device=self.device,
        )
        action_batch = torch.tensor(
            batch.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.array(batch.next_state),
            dtype=torch.float32,
            device=self.device,
        )
        done_batch = torch.tensor(
            batch.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # >>> reset noise for both policy & target <<<
        if hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, "reset_noise"):
            self.target_net.reset_noise()
        # =============================
        # 2) Q(s,a) and target (Double DQN)
        # =============================

        # Q(s,a) from policy_net
        q_values = self.policy_net(state_batch).gather(1, action_batch)  # (B,1)

        with torch.no_grad():
            # online net chose best action at s'
            next_q_online = self.policy_net(next_state_batch)             # (B,A)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)     # (B,1)

            # target net evaluates Q for that action
            next_q_target = self.target_net(next_state_batch).gather(1, next_actions)

            target = reward_batch + self.gamma * next_q_target * (1.0 - done_batch)

        # TD-error
        td_errors = target - q_values  # (B,1)

        # =============================
        # 3) Loss (PER: weighted)
        # =============================
        if self.use_per and weights is not None:
            # Weighted SmoothL1Loss
            loss = (weights * torch.nn.functional.smooth_l1_loss(
                q_values, target, reduction="none"
            )).mean()
        else:
            loss = nn.SmoothL1Loss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # =============================
        # 4) Update priority & beta
        # =============================
        if self.use_per and indices is not None:
            # prio_i = |TD error| + eps
            prios = td_errors.detach().abs().cpu().numpy().squeeze(1) + self.per_eps
            if isinstance(self.buffer, PrioritizedReplayBuffer):
                self.buffer.update_priorities(indices, prios)

            # anneal beta gradually to 1.0
            self.per_beta = min(
                1.0,
                self.per_beta + (1.0 - self.per_beta_start) / max(1, self.per_beta_frames)
            )

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


    def decay_epsilon(self):
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        pass


# =============================
# 5. TRAINING LOOP (ONE ENV)
# =============================

def train_dqn_routing(
    env: UAVRoutingEnvTorch,
    num_episodes: int = 2000,
    gamma: float = 0.95,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_capacity: int = 100_000,
    target_update_freq: int = 500,
    device: str = "cpu",
    log_interval: int = 500,

    use_gnn: bool = False,
    graph_tensors=None,   # (node_feats, edge_index, edge_attr)
):
    """
    Train DQN on a routing environment (1 time_slot).

    Logging:
      - Count episodes ended by each QoS reason:
        + reached_goal_qos_ok
        + reached_goal_qos_bad
        + e2e_loss_exceeded
        + delay_exceeded
        + jitter_exceeded
        + max_hops_exceeded
        + invalid_move
      - Calculate avg cum_delay / cum_jitter / e2e_loss / hops for success/fail.
    """
    device = torch.device(device)
    if use_gnn:
        assert graph_tensors is not None, "graph_tensors must be provided when use_gnn=True"
        node_feats, edge_index, edge_attr = graph_tensors
    else:
        node_feats = edge_index = edge_attr = None

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        device=device,

        use_per=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100_000,

        
        use_gnn=use_gnn,
        graph_tensors=graph_tensors,
    )

    # ---- statistics for logging ----
    term_counts = {
        "reached_goal_qos_ok": 0,
        "reached_goal_qos_bad": 0,
        "e2e_loss_exceeded": 0,
        "delay_exceeded": 0,
        "jitter_exceeded": 0,
        "max_hops_exceeded": 0,
        "invalid_move": 0,
        "other": 0,
    }

    # accumulate QoS at the end of episode for success/fail
    success_stats = {
        "count": 0,
        "cum_delay": 0.0,
        "cum_jitter": 0.0,
        "e2e_loss": 0.0,
        "hops": 0.0,
    }
    fail_stats = {
        "count": 0,
        "cum_delay": 0.0,
        "cum_jitter": 0.0,
        "e2e_loss": 0.0,
        "hops": 0.0,
    }

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        last_info = None

        while not done:
            valid_actions = env.get_valid_actions()
            # TRAIN: explore=True
            action_idx = agent.select_action(state, valid_actions=valid_actions, explore=True)


            next_state, reward, done, info = env.step(action_idx)

            agent.store_transition(state, action_idx, reward, next_state, done)
            loss = agent.update()

            state = next_state
            total_reward += reward
            last_info = info

        # ----- logging episode-level -----
        reason = last_info.get("info") if last_info is not None else "other"
        if reason not in term_counts:
            reason = "other"
        term_counts[reason] += 1

        # QoS last episode
        cum_delay = last_info.get("cum_delay", 0.0) if last_info else 0.0
        cum_jitter = last_info.get("cum_jitter", 0.0) if last_info else 0.0
        e2e_loss = last_info.get("e2e_loss", 0.0) if last_info else 0.0
        hops = last_info.get("hop", 0) if last_info else 0

        if reason == "reached_goal_qos_ok":
            success_stats["count"] += 1
            success_stats["cum_delay"] += cum_delay
            success_stats["cum_jitter"] += cum_jitter
            success_stats["e2e_loss"] += e2e_loss
            success_stats["hops"] += hops
        else:
            fail_stats["count"] += 1
            fail_stats["cum_delay"] += cum_delay
            fail_stats["cum_jitter"] += cum_jitter
            fail_stats["e2e_loss"] += e2e_loss
            fail_stats["hops"] += hops

        # ----- logging episode-level -----
        if ep % log_interval == 0 or ep == num_episodes:
            print(
                f"[DQN] Episode {ep}/{num_episodes}, "
                f"reward={total_reward:.2f}, epsilon={agent.epsilon:.3f}"
            )

            total_episodes = sum(term_counts.values())
            if total_episodes == 0:
                total_episodes = 1  # tránh chia 0

            print("  Termination reasons (from start):")
            for k, v in term_counts.items():
                pct = 100.0 * v / total_episodes
                print(f"    - {k:22s}: {v:5d} ({pct:5.1f}%)")

            if success_stats["count"] > 0:
                c = success_stats["count"]
                print("  Success QoS (reached_goal_qos_ok):")
                print(
                    f"    avg cum_delay  = {success_stats['cum_delay'] / c:.2f} ms, "
                    f"avg cum_jitter = {success_stats['cum_jitter'] / c:.2f} ms"
                )
                print(
                    f"    avg e2e_loss   = {success_stats['e2e_loss'] / c:.3f}, "
                    f"avg hops      = {success_stats['hops'] / c:.2f}"
                )
            else:
                print("  Success QoS (reached_goal_qos_ok): no episodes yet.")

            if fail_stats["count"] > 0:
                c = fail_stats["count"]
                print("  Fail QoS / khác:")
                print(
                    f"    avg cum_delay  = {fail_stats['cum_delay'] / c:.2f} ms, "
                    f"avg cum_jitter = {fail_stats['cum_jitter'] / c:.2f} ms"
                )
                print(
                    f"    avg e2e_loss   = {fail_stats['e2e_loss'] / c:.3f}, "
                    f"avg hops      = {fail_stats['hops'] / c:.2f}"
                )
            print("-" * 60)

    return agent



def rollout_dqn_episode(
    env: UAVRoutingEnvTorch,
    agent: DQNAgent,
    device: str = "cpu",
    max_steps: int | None = None,
):
    """
    
    """
    device = torch.device(device)

    if max_steps is None:
        max_steps = env.max_hops + 5

    # Save current training/eval state
    was_training = agent.policy_net.training

    # Set policy to eval mode: NoisyLinear will use only μ (no noise)
    agent.policy_net.eval()

    state = env.reset()
    total_dist = 0.0
    total_delay = 0.0
    path_indices = [env.start_idx]

    visited = set([env.start_idx])
    done = False

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # EVAL: explore=False => NO reset_noise
        action_idx = agent.select_action(
            state, valid_actions=valid_actions, explore=False
        )

        # avoid simple loops
        if action_idx in visited and action_idx != env.goal_idx:
            pruned = [a for a in valid_actions if a not in visited]
            if pruned:
                action_idx = agent.select_action(
                    state, valid_actions=pruned, explore=False
                )

        key = (env.current_idx, action_idx)
        if key not in env.edge_dict:
            break

        dist, delay_ms, bw, jitter_ms, loss = env.edge_dict[key]
        total_dist += dist
        total_delay += delay_ms

        next_state, reward, done, info = env.step(action_idx)
        path_indices.append(action_idx)
        visited.add(action_idx)
        state = next_state

        if done:
            break

    # Restore original mode
    if was_training:
        agent.policy_net.train()

    if env.current_idx != env.goal_idx:
        return None, None, None

    path_node_ids = env.decode_path(path_indices)
    return path_node_ids, total_dist, total_delay




def extract_greedy_path_dqn(
    env: UAVRoutingEnvTorch,
    agent: DQNAgent,
    device: str = "cpu",
    num_trials: int = 10,
    epsilon_eval: float = 0.05,
    max_steps: int | None = None,
    verbose: bool = False,
):
    """
    Extract the best path found by running multiple evaluation episodes
    """
    best_path = None
    best_dist = None
    best_delay = None

    for i in range(num_trials):
        path, dist, delay = rollout_dqn_episode(
            env,
            agent,
            device=device,
            max_steps=max_steps,
        )

        if path is None:
            continue

        if best_path is None or delay < best_delay:
            best_path = path
            best_dist = dist
            best_delay = delay

        if verbose:
            print(f"[EVAL TRIAL {i+1}/{num_trials}] path={path}, dist={dist}, delay={delay} ms")

    if best_path is None and verbose:
        print("[EVAL] No path found to BS2 in num_trials.")

    return best_path, best_dist, best_delay



def build_graph_from_env(env: UAVRoutingEnvTorch, device: str = "cpu"):
    """
    Build PyG-style graph tensors from UAVRoutingEnvTorch.

    Returns:
      - node_feats:  (N, F_node)  float32
      - edge_index: (2, E)       long
      - edge_attr:  (E, F_edge)  float32
    """
    device = torch.device(device)
    num_nodes = env.num_nodes

    # -------------------------
    # 1) NODE FEATURES
    # -------------------------
    # coords: (N, 2) = (lat, lon)
    coords = env.coords.astype(np.float32)          # (N, 2)
    lat = coords[:, 0]
    lon = coords[:, 1]

    # Normalize coordinates: (x - mean) / std
    lat_mean, lat_std = lat.mean(), (lat.std() + 1e-6)
    lon_mean, lon_std = lon.mean(), (lon.std() + 1e-6)

    lat_norm = (lat - lat_mean) / lat_std
    lon_norm = (lon - lon_mean) / lon_std

    # dist_to_goal is already in env, normalize to [0, 1]
    dist_to_goal = env.dist_to_goal.astype(np.float32)        # (N,)
    max_d = max(env.max_dist_to_goal, 1e-6)
    dist_to_goal_norm = dist_to_goal / max_d

    # one-hot for start / goal
    is_start = np.zeros(num_nodes, dtype=np.float32)
    is_goal = np.zeros(num_nodes, dtype=np.float32)
    is_start[env.start_idx] = 1.0
    is_goal[env.goal_idx] = 1.0

    # Combine node features
    # node_feat = [lat_norm, lon_norm, dist_to_goal_norm, is_start, is_goal]
    node_feats_np = np.stack(
        [lat_norm, lon_norm, dist_to_goal_norm, is_start, is_goal],
        axis=1
    ).astype(np.float32)   # (N, 5)

    # -------------------------
    # 2) EDGE INDEX + EDGE FEATURES
    # -------------------------
    src_list = []
    dst_list = []
    edge_feats = []   # each element is [dist_norm, delay_norm, bw_norm, jitter_norm, loss_norm]

    # Normalize according to QoS scale in env
    max_delay = max(env.max_delay_ms, 1e-6)
    max_jitter = max(env.max_jitter_ms, 1e-6)
    target_bw = max(env.target_bw_mbps, 1e-6)
    max_loss = max(env.max_loss, 1e-6) if (env.max_loss is not None and env.max_loss > 0) else 1.0

    # You can also compute max_dist_edge if you want to normalize distance;
    # here we just use max_dist_to_goal for simplicity.
    max_dist_edge = max_d

    for (u_idx, v_idx), (dist, delay_ms, bw, jitter_ms, loss) in env.edge_dict.items():
        # forward edge u->v
        src_list.append(u_idx)
        dst_list.append(v_idx)

        dist_norm   = float(dist)       / max_dist_edge
        delay_norm  = float(delay_ms)   / max_delay
        bw_norm     = float(bw)         / target_bw
        jitter_norm = float(jitter_ms)  / max_jitter
        loss_norm   = float(loss)       / max_loss

        edge_feats.append([dist_norm, delay_norm, bw_norm, jitter_norm, loss_norm])


    if len(src_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 5), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(
            [src_list, dst_list],
            dtype=torch.long,
            device=device,
        )                         # (2, E)
        edge_attr = torch.tensor(
            np.array(edge_feats, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )                         # (E, 5)

    # -------------------------
    # 3) Convert node_feats sang tensor
    # -------------------------
    node_feats = torch.tensor(node_feats_np, dtype=torch.float32, device=device)  # (N, 5)

    return node_feats, edge_index, edge_attr



def build_nx_graph_from_env(env: UAVRoutingEnvTorch) -> nx.DiGraph:
    """
    Build networkx.DiGraph from UAVRoutingEnvTorch.
    """
    G = nx.DiGraph()

    # Add nodes
    for idx, nid in enumerate(env.node_ids):
        lat, lon = env.coords[idx]
        G.add_node(
            idx,
            label=nid,
            pos=(float(lon), float(lat)),  # (x=lon, y=lat)
        )

    # Add edges
    for (u_idx, v_idx), (dist, delay_ms, bw, jitter_ms, loss) in env.edge_dict.items():
        G.add_edge(
            u_idx,
            v_idx,
            dist=float(dist),
            delay_ms=float(delay_ms),
            bw=float(bw),
            jitter_ms=float(jitter_ms),
            loss=float(loss),
        )

    return G


def visualize_env_graph_simple(
    env: UAVRoutingEnvTorch,
    color_by: str = "bw",   
    figsize=(8, 6),
    title: str | None = None,
    save_path: str | None = None,
    show_labels: bool = True,
):


    G = build_nx_graph_from_env(env)

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.get_node_attributes(G, "pos")
    labels = nx.get_node_attributes(G, "label")

    node_colors = []
    node_sizes = []
    for idx in G.nodes():
        if idx == env.start_idx:
            node_colors.append("green")
            node_sizes.append(400)
        elif idx == env.goal_idx:
            node_colors.append("red")
            node_sizes.append(400)
        else:
            node_colors.append("lightblue")
            node_sizes.append(200)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="black",
        ax=ax,
    )

    if show_labels:
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    # --- Edge color by metric ---
    edges = list(G.edges(data=True))
    if color_by == "bw":
        values = np.array([data["bw"] for (_, _, data) in edges], dtype=float)
        cmap = plt.cm.Blues
        metric_label = "Bandwidth (Mbps)"
    elif color_by == "delay":
        values = np.array([data["delay_ms"] for (_, _, data) in edges], dtype=float)
        cmap = plt.cm.OrRd
        metric_label = "Delay (ms)"
    else:
        values = None
        cmap = None
        metric_label = ""

    if values is not None and len(values) > 0:
        vmin, vmax = float(values.min()), float(values.max())
        if vmin == vmax:
            vmin, vmax = 0.0, vmin + 1.0
        edge_colors = [cmap((v - vmin) / (vmax - vmin)) for v in values]
    else:
        edge_colors = "gray"

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        arrows=True,
        width=1.2,
        alpha=0.8,
        arrowsize=10,
        ax=ax,
    )

    if values is not None and len(values) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(values)
        cbar = plt.colorbar(sm, ax=ax)     
        cbar.set_label(metric_label)

    if title is None:
        title = f"UAV Routing Graph (color_by={color_by})"

    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[VIS] Saved graph to {save_path}")

    plt.show()


def graph_features_to_tables(
    env: UAVRoutingEnvTorch,
    node_feats: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    time_slot: int | None = None,
    save_dir: str | None = None,
    node_feat_names: list[str] | None = None,
    edge_feat_names: list[str] | None = None,
):
    """
    
    """


    if isinstance(node_feats, torch.Tensor):
        node_feats_np = node_feats.detach().cpu().numpy()
    else:
        node_feats_np = np.asarray(node_feats)

    if isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.detach().cpu().numpy()
    else:
        edge_index_np = np.asarray(edge_index)

    if isinstance(edge_attr, torch.Tensor):
        edge_attr_np = edge_attr.detach().cpu().numpy()
    else:
        edge_attr_np = np.asarray(edge_attr)

    N, F = node_feats_np.shape
    _, E = edge_index_np.shape

    # ------------------------
    # 1) NODE FEATURE TABLE
    # ------------------------
    node_rows = []
    for idx in range(N):
        node_id = env.idx2id[idx]
        lat, lon = env.coords[idx]
        dist_to_goal = float(env.dist_to_goal[idx])
        dist_to_goal_norm = dist_to_goal / max(env.max_dist_to_goal, 1e-6)

        # Node type: BS1 / BS2 / UAV (depending on how the id is named)
        if node_id == env.start_id:
            node_type = "BS_START"
        elif node_id == env.goal_id:
            node_type = "BS_GOAL"
        elif node_id.startswith("BS"):
            node_type = "BS_OTHER"
        else:
            node_type = "UAV"

        base_info = {
            "node_idx": idx,
            "node_id": node_id,
            "node_type": node_type,
            "lat": lat,
            "lon": lon,
            "dist_to_goal": dist_to_goal,
            "dist_to_goal_norm": dist_to_goal_norm,
        }

        # Add each feature from node_feats
        for f_idx in range(F):
            col_name = (
                node_feat_names[f_idx]
                if node_feat_names is not None and f_idx < len(node_feat_names)
                else f"feat_{f_idx}"
            )
            base_info[col_name] = node_feats_np[idx, f_idx]

        node_rows.append(base_info)

    node_df = pd.DataFrame(node_rows)

    # ------------------------
    # 2) EDGE FEATURE TABLE
    # ------------------------
    edge_rows = []
    for e in range(E):
        u_idx = int(edge_index_np[0, e])
        v_idx = int(edge_index_np[1, e])

        u_id = env.idx2id[u_idx]
        v_id = env.idx2id[v_idx]

        # if env.edge_dict has raw QoS, also extract it for inspection
        if (u_idx, v_idx) in env.edge_dict:
            dist, delay_ms, bw, jitter_ms, loss = env.edge_dict[(u_idx, v_idx)]
        else:
            dist = delay_ms = bw = jitter_ms = loss = None

        base_info = {
            "edge_idx": e,
            "u_idx": u_idx,
            "v_idx": v_idx,
            "u_id": u_id,
            "v_id": v_id,
            "dist_raw": dist,
            "delay_ms_raw": delay_ms,
            "bw_mbps_raw": bw,
            "jitter_ms_raw": jitter_ms,
            "loss_raw": loss,
        }

        # Add each feature from edge_attr
        D = edge_attr_np.shape[1]
        for d in range(D):
            col_name = (
                edge_feat_names[d]
                if edge_feat_names is not None and d < len(edge_feat_names)
                else f"efeat_{d}"
            )
            base_info[col_name] = edge_attr_np[e, d]

        edge_rows.append(base_info)

    edge_df = pd.DataFrame(edge_rows)

    # ------------------------
    # 3) Save files if needed
    # ------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        ts_str = f"{time_slot:04d}" if time_slot is not None else "all"

        node_csv = os.path.join(save_dir, f"nodes_t{ts_str}.csv")
        edge_csv = os.path.join(save_dir, f"edges_t{ts_str}.csv")

        node_df.to_csv(node_csv, index=False)
        edge_df.to_csv(edge_csv, index=False)
        print(f"[DEBUG] Saved node features to {node_csv}")
        print(f"[DEBUG] Saved edge features to {edge_csv}")

    return node_df, edge_df